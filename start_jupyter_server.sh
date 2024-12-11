








window_index=(1 2 3 4 5 6 7 8 9)
tries=(0)
nsigs=(0)

for nsig in ${nsigs[@]}; do
    for window in ${window_index[@]}; do
        for try in ${tries[@]}; do
            mkdir -p results/marie_samples/window_scan_herwig_extended2_${nsig}/window_${window}/try_${try}
            cp  results/window_scan_herwig_extended2_${nsig}/window_${window}/try_${try}/samples_CR.npy results/marie_samples/window_scan_herwig_extended2_${nsig}/window_${window}/try_${try}/
            cp  results/window_scan_herwig_extended2_${nsig}/window_${window}/try_${try}/samples_SR.npy results/marie_samples/window_scan_herwig_extended2_${nsig}/window_${window}/try_${try}/
        done
    done
done

# cp -r results/marie_samples/window_scan_ext1_0 /global/cfs/cdirs/m4539/ranit/
# cp -r results/marie_samples/window_scan_ext1_1000 /global/cfs/cdirs/m4539/ranit/


