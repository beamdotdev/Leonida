// ID3D12Device3::OpenExistingHeapFromAddress, OpenExistingHeapFromFileMapping and ID3D12Device5::CreateLifetimeTracker
// on arguments the host D3D12 runtime refuses (null address, a file mapping that is not a handle, a null owner), all
// E_INVALIDARG on the RTX 4080 SUPER, Intel UHD 770 and WARP. An existing-heap call the host accepts is not compared
// (this device reports EXISTING_HEAPS unsupported). A lifetime tracker for a valid owner is: the host takes no
// reference to the owner and never notifies it, refuses a foreign IID with E_NOINTERFACE, and DestroyOwnedObject
// refuses every object the public API makes with E_INVALIDARG (RTX 3070 / 621.01 and WARP). Checked on the
// Vulkan-backed device and, when present, on the host devices too. No shaders or game data.
// https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12device3-openexistingheapfromaddress
#include "vk_test_device.h"
#include "host_adapter.h"
#include "host_wine.h"
#include "../shims/d3d12_com.h"
#include <dxgi1_4.h>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "%s line %d: %s\n", who, __LINE__, #condition); return 1; } \
} while (0)
#define EXPECT_HR(expr, want) do { \
    const HRESULT got_ = (expr); \
    if (got_ != (want)) { \
        std::fprintf(stderr, "%s line %d: %s = 0x%08lX, expected 0x%08lX\n", who, __LINE__, #expr, \
                     static_cast<unsigned long>(got_), static_cast<unsigned long>(want)); \
        return 1; \
    } \
} while (0)

namespace {
template<class T>
struct Ref {
    T* p = nullptr;
    ~Ref() { if (p) p->Release(); }
    T* operator->() const { return p; }
};

// Counts its references and LifetimeStateUpdated calls; never freed, so a stray Release cannot fault.
struct Owner final : ID3D12LifetimeOwner {
    LONG refs = 1;
    int calls = 0;
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** out) override {
        if (!out) return E_POINTER;
        *out = nullptr;
        if (riid != __uuidof(IUnknown) && riid != __uuidof(ID3D12LifetimeOwner)) return E_NOINTERFACE;
        *out = this;
        AddRef();
        return S_OK;
    }
    ULONG STDMETHODCALLTYPE AddRef() override { return ULONG(InterlockedIncrement(&refs)); }
    ULONG STDMETHODCALLTYPE Release() override { return ULONG(InterlockedDecrement(&refs)); }
    void STDMETHODCALLTYPE LifetimeStateUpdated(D3D12_LIFETIME_STATE) override { ++calls; }
};

int check(ID3D12Device* device, const char* who) {
    Ref<ID3D12Device3> device3;
    EXPECT(SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(&device3.p))));
    Ref<ID3D12Heap> heap;
    EXPECT_HR(device3->OpenExistingHeapFromAddress(nullptr, IID_PPV_ARGS(&heap.p)), E_INVALIDARG);
    EXPECT(heap.p == nullptr);
    EXPECT_HR(device3->OpenExistingHeapFromFileMapping(nullptr, IID_PPV_ARGS(&heap.p)), E_INVALIDARG);
    EXPECT_HR(device3->OpenExistingHeapFromFileMapping(reinterpret_cast<HANDLE>(0x1234), IID_PPV_ARGS(&heap.p)),
              E_INVALIDARG);
    EXPECT(heap.p == nullptr);
    Ref<ID3D12Device5> device5;
    if (SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(&device5.p)))) {
        Ref<ID3D12LifetimeTracker> tracker;
        EXPECT_HR(device5->CreateLifetimeTracker(nullptr, IID_PPV_ARGS(&tracker.p)), E_INVALIDARG);
        EXPECT(tracker.p == nullptr);

        // A valid owner: the host hands out a tracker without referencing or notifying the owner, refuses an IID
        // that is not the tracker's, and DestroyOwnedObject refuses every object the public API can make.
        Owner owner;
        Ref<ID3D12Heap> wrong;
        EXPECT_HR(device5->CreateLifetimeTracker(&owner, IID_PPV_ARGS(&wrong.p)), E_NOINTERFACE);
        EXPECT(wrong.p == nullptr);
        EXPECT_HR(device5->CreateLifetimeTracker(&owner, IID_PPV_ARGS(&tracker.p)), S_OK);
        EXPECT(tracker.p != nullptr);
        EXPECT(owner.refs == 1 && owner.calls == 0);
        Ref<ID3D12Device> back;
        EXPECT_HR(tracker->GetDevice(IID_PPV_ARGS(&back.p)), S_OK);
        Ref<ID3D12Pageable> pageable;
        EXPECT_HR(tracker->QueryInterface(IID_PPV_ARGS(&pageable.p)), E_NOINTERFACE);
        Ref<ID3D12Fence> fence;
        EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));
        fence->AddRef();
        const ULONG fence_refs = fence->Release();
        EXPECT_HR(tracker->DestroyOwnedObject(fence.p), E_INVALIDARG);
        fence->AddRef();
        EXPECT(fence->Release() == fence_refs);
        EXPECT_HR(tracker->DestroyOwnedObject(tracker.p), E_INVALIDARG);
        Ref<ID3D12LifetimeTracker> other;
        EXPECT_HR(device5->CreateLifetimeTracker(&owner, IID_PPV_ARGS(&other.p)), S_OK);
        EXPECT_HR(tracker->DestroyOwnedObject(other.p), E_INVALIDARG);
        EXPECT(owner.refs == 1 && owner.calls == 0);
    }
    std::printf("%s: existing-heap and lifetime-tracker refusals match the host runtime\n", who);
    return 0;
}
}  // namespace

int main() {
    vktest::Device vk;
    const auto status = vk.initialize();
    if (status == vktest::DeviceStatus::unavailable) return 77;
    if (status != vktest::DeviceStatus::ready) return 1;
    int failures = 0;
    {
        Ref<ID3D12Device> ours;
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = vk.instance_proc;
        info.instance = vk.instance;
        info.physical = vk.physical;
        info.device = vk.device;
        info.queue = vk.queue;
        info.queue_family = vk.family;
        if (FAILED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&ours.p)))) return 1;
        failures += check(ours.p, "vulkan");
        Ref<IDXGIFactory4> factory;
        if (!hostwine::host_is_wine() && SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p)))) {
            for (const bool warp : {false, true}) {
                Ref<IDXGIAdapter> adapter;
                Ref<ID3D12Device> host;
                std::string adapter_name;
                if (!hostadapter::reference(factory.p, vk, warp, &adapter.p, adapter_name)) continue;
                if (FAILED(D3D12CreateDevice(adapter.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&host.p)))) continue;
                failures += check(host.p, ("host " + adapter_name).c_str());
            }
        }
    }
    if (vk.validation_errors()) {
        std::fprintf(stderr, "%u validation errors\n", vk.validation_errors());
        return 1;
    }
    return failures ? 1 : 0;
}
