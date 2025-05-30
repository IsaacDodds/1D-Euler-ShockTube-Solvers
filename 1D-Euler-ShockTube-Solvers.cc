//===============================//
//       Header Inclusions-Main is at bottom     //
//===============================//
#include <stdio.h>     // Standard input/output
#include <stdlib.h>    // Memory allocation
#include <math.h>      // Math operations like sqrt()
#include <omp.h>       // OpenMP for parallel computing
#include <time.h>   // For clock()

//===============================//
//     Global Constants         //
//===============================//
#define N 100           // Number of grid points
#define GAMMA 1.4       // Adiabatic index (polytropic)
#define DX (1.0 / N)    // Grid spacing (assuming unit domain)
#define T_MAX 0.2       // Final time for Problem A
#define T_MAX2 0.15     // Final time for Problem B

//===============================//
//     Global Pointers          //
//===============================//

// Primary mesh and hydrodynamic variables
double *x, *rho, *v, *p, *e, *E, *rhov;

// Arrays for updated values during each timestep
double *newrho, *newrhov, *newE;

// Speed of sound
double *cs;

// Fluxes at cell interfaces
double *flux_rho, *flux_rhov, *flux_E;

// Lax-Friedrichs intermediate fluxes
double *halfflux_rho, *halfflux_rhov, *halfflux_E;

// Predictor states for Lax-Wendroff
double *half_rho, *half_rhov, *half_E;
double *half_v, *half_p, *half_e, *half_cs;
double *half_flux_rho, *half_flux_rhov, *half_flux_E;

// HLL left/right states at interfaces
double *rhoL, *rhoR, *rhovL, *rhovR;
double *EL, *ER, *pL, *pR;
double *vL, *vR, *eL, *eR, *csL, *csR;
double *flux_rhoL, *flux_rhoR;
double *flux_rhovL, *flux_rhovR;
double *flux_EL, *flux_ER;

// Tracer for contact discontinuity tracking
double *tracer, *newtracer, *flux_tracer, *halfflux_tracer;

//===============================//
//      Memory Allocation       //
//===============================//
void allocate_all(int order) {
    int size = N + 2 * order * order;

    // Allocate all simulation arrays
    x = malloc(size * sizeof(double));
    rho = malloc(size * sizeof(double));
    rhov = malloc(size * sizeof(double));
    E = malloc(size * sizeof(double));
    v = malloc(size * sizeof(double));
    p = malloc(size * sizeof(double));
    e = malloc(size * sizeof(double));
    cs = malloc(size * sizeof(double));

    newrho = malloc(size * sizeof(double));
    newrhov = malloc(size * sizeof(double));
    newE = malloc(size * sizeof(double));

    flux_rho = malloc(size * sizeof(double));
    flux_rhov = malloc(size * sizeof(double));
    flux_E = malloc(size * sizeof(double));

    halfflux_rho = malloc(size * sizeof(double));
    halfflux_rhov = malloc(size * sizeof(double));
    halfflux_E = malloc(size * sizeof(double));

    half_rho = malloc(size * sizeof(double));
    half_rhov = malloc(size * sizeof(double));
    half_E = malloc(size * sizeof(double));
    half_v = malloc(size * sizeof(double));
    half_p = malloc(size * sizeof(double));
    half_e = malloc(size * sizeof(double));
    half_cs = malloc(size * sizeof(double));
    half_flux_rho = malloc(size * sizeof(double));
    half_flux_rhov = malloc(size * sizeof(double));
    half_flux_E = malloc(size * sizeof(double));

    rhoL = malloc(size * sizeof(double));
    rhoR = malloc(size * sizeof(double));
    rhovL = malloc(size * sizeof(double));
    rhovR = malloc(size * sizeof(double));
    EL = malloc(size * sizeof(double));
    ER = malloc(size * sizeof(double));
    pL = malloc(size * sizeof(double));
    pR = malloc(size * sizeof(double));
    vL = malloc(size * sizeof(double));
    vR = malloc(size * sizeof(double));
    eL = malloc(size * sizeof(double));
    eR = malloc(size * sizeof(double));
    csL = malloc(size * sizeof(double));
    csR = malloc(size * sizeof(double));
    flux_rhoL = malloc(size * sizeof(double));
    flux_rhoR = malloc(size * sizeof(double));
    flux_rhovL = malloc(size * sizeof(double));
    flux_rhovR = malloc(size * sizeof(double));
    flux_EL = malloc(size * sizeof(double));
    flux_ER = malloc(size * sizeof(double));

    tracer = malloc(size * sizeof(double));
    newtracer = malloc(size * sizeof(double));
    flux_tracer = malloc(size * sizeof(double));
    halfflux_tracer = malloc(size * sizeof(double));
}


//===============================//
//       Memory Deallocation    //
//===============================//
void free_all() {
    // Free every allocated pointer to avoid memory leaks
    free(x); free(rho); free(rhov); free(E);
    free(v); free(p); free(e); free(cs);

    free(newrho); free(newrhov); free(newE);
    free(flux_rho); free(flux_rhov); free(flux_E);
    free(halfflux_rho); free(halfflux_rhov); free(halfflux_E);

    free(half_rho); free(half_rhov); free(half_E);
    free(half_v); free(half_p); free(half_e); free(half_cs);
    free(half_flux_rho); free(half_flux_rhov); free(half_flux_E);

    free(rhoL); free(rhoR); free(rhovL); free(rhovR);
    free(EL); free(ER); free(pL); free(pR);
    free(vL); free(vR); free(eL); free(eR);
    free(csL); free(csR);

    free(flux_rhoL); free(flux_rhoR);
    free(flux_rhovL); free(flux_rhovR);
    free(flux_EL); free(flux_ER);

    free(tracer); free(newtracer);
    free(flux_tracer); free(halfflux_tracer);
}


void setup_sod_problemA(double *rho, double *rhov, double *E) {
    #pragma omp parallel for
    for (int i = 0; i <= N + 1; i++) {
        x[i] = (i - 0.5) * DX;

        double local_rho, local_v, local_p;
        if (x[i] < 0.3) {
            local_rho = 1.0;
            local_v = 0.75;
            local_p = 1.0;
        } else {
            local_rho = 0.125;
            local_v = 0.0;
            local_p = 0.1;
        }

        rho[i] = local_rho;
        v[i] = local_v;
        p[i] = local_p;
        E[i] = local_p / (GAMMA - 1.0) + 0.5 * local_rho * local_v * local_v;
        rhov[i] = local_rho * local_v;
        tracer[i] = (x[i] < 0.3) ? 1.0 : 0.0;
    }
}

void setup_sod_problemB(double *rho, double *rhov, double *E) {
    #pragma omp parallel for
    for (int i = 0; i <= N + 1; i++) {
        x[i] = (i - 0.5) * DX;

        double local_rho = 1.0;
        double local_v, local_p = 0.4;

        if (x[i] < 0.5) {
            local_v = -2.0;
        } else {
            local_v = 2.0;
        }

        rho[i] = local_rho;
        v[i] = local_v;
        p[i] = local_p;
        E[i] = local_p / (GAMMA - 1.0) + 0.5 * local_rho * local_v * local_v;
        rhov[i] = local_rho * local_v;
    }
}


void boundary_problemA1st(double *rho, double *rhov, double *E) {
    rho[0] = rho[1];     rho[N + 1] = rho[N];
    rhov[0] = rhov[1];   rhov[N + 1] = rhov[N];
    E[0] = E[1];         E[N + 1] = E[N];
}


void boundary_problemB1st(double *rho, double *v, double *p, double *e, double *E, double *rhov) {
    // Left boundary (copy from interior)
    rho[0] = rho[1];
    v[0] = v[1];
    p[0] = p[1];
    e[0] = e[1];

    rhov[0] = rho[0] * v[0];
    E[0] = rho[0] * e[0] + 0.5 * rho[0] * v[0] * v[0];

    // Right boundary (copy from interior)
    rho[N + 1] = rho[N];
    v[N + 1] = v[N];
    p[N + 1] = p[N];
    e[N + 1] = e[N];

    rhov[N + 1] = rho[N + 1] * v[N + 1];
    E[N + 1] = rho[N + 1] * e[N + 1] + 0.5 * rho[N + 1] * v[N + 1] * v[N + 1];
}

void compute_observables(double *rho, double *rhov, double *E, double *cs, double *v, double *p, double *e) {
    #pragma omp parallel for
    for (int i = 0; i <= N + 1; i++) {
        v[i] = rhov[i] / rho[i];
        p[i] = (GAMMA - 1.0) * (E[i] - 0.5 * rho[i] * v[i] * v[i]);
        e[i] = E[i] / rho[i] - 0.5 * v[i] * v[i];
        cs[i] = sqrt(GAMMA * p[i] / rho[i]);
    }
}


void compute_fluxes(double *rho, double *rhov, double *E, double *v, double *p, double *flux_rho, double *flux_rhov, double *flux_E) {
    #pragma omp parallel for
    for (int i = 0; i <= N + 1; i++) {
        flux_rho[i] = rhov[i];
        flux_rhov[i] = rhov[i] * v[i] + p[i];
        flux_E[i] = v[i] * (E[i] + p[i]);
        flux_tracer[i] = tracer[i] * v[i];
    }
}


void lax_friedrichs_flux(double dt) {
    #pragma omp parallel for
    for (int i = 0; i <= N; i++) {
        halfflux_rho[i] = 0.5 * (flux_rho[i] + flux_rho[i + 1]) - 0.5 * DX / dt * (rho[i + 1] - rho[i]);
        halfflux_rhov[i] = 0.5 * (flux_rhov[i] + flux_rhov[i + 1]) - 0.5 * DX / dt * (rhov[i + 1] - rhov[i]);
        halfflux_E[i] = 0.5 * (flux_E[i] + flux_E[i + 1]) - 0.5 * DX / dt * (E[i + 1] - E[i]);
        halfflux_tracer[i] = 0.5 * (flux_tracer[i] + flux_tracer[i + 1]) - 0.5 * DX / dt * (tracer[i + 1] - tracer[i]);
    }
}


void Lax_W_step(double dt) {
    double *rho_pred = malloc((N + 2) * sizeof(double));
    double *rhov_pred = malloc((N + 2) * sizeof(double));
    double *E_pred = malloc((N + 2) * sizeof(double));
    double *v_pred = malloc((N + 2) * sizeof(double));
    double *p_pred = malloc((N + 2) * sizeof(double));

    // Midpoint predictor
    #pragma omp parallel for
    for (int i = 0; i <= N; i++) {
        rho_pred[i] = 0.5 * (rho[i] + rho[i + 1]) + 0.5 * dt / DX * (flux_rho[i] - flux_rho[i + 1]);
        rhov_pred[i] = 0.5 * (rhov[i] + rhov[i + 1]) + 0.5 * dt / DX * (flux_rhov[i] - flux_rhov[i + 1]);
        E_pred[i] = 0.5 * (E[i] + E[i + 1]) + 0.5 * dt / DX * (flux_E[i] - flux_E[i + 1]);
    }

    // Fluxes at predicted state
    #pragma omp parallel for
    for (int i = 0; i <= N; i++) {
        v_pred[i] = rhov_pred[i] / rho_pred[i];
        p_pred[i] = (GAMMA - 1.0) * (E_pred[i] - 0.5 * rho_pred[i] * v_pred[i] * v_pred[i]);

        halfflux_rho[i] = rhov_pred[i];
        halfflux_rhov[i] = rhov_pred[i] * v_pred[i] + p_pred[i];
        halfflux_E[i] = v_pred[i] * (E_pred[i] + p_pred[i]);
    }

    // Update step
    #pragma omp parallel for
    for (int i = 1; i <= N; i++) {
        newrho[i] = rho[i] - dt / DX * (halfflux_rho[i] - halfflux_rho[i - 1]);
        newrhov[i] = rhov[i] - dt / DX * (halfflux_rhov[i] - halfflux_rhov[i - 1]);
        newE[i] = E[i] - dt / DX * (halfflux_E[i] - halfflux_E[i - 1]);
    }

    free(rho_pred); free(rhov_pred); free(E_pred); free(v_pred); free(p_pred);
}


void euler_step(double dt) {
    #pragma omp parallel for
    for (int i = 1; i <= N; i++) {
        newrho[i] = rho[i] - dt / DX * (halfflux_rho[i] - halfflux_rho[i - 1]);
        newrhov[i] = rhov[i] - dt / DX * (halfflux_rhov[i] - halfflux_rhov[i - 1]);
        newE[i] = E[i] - dt / DX * (halfflux_E[i] - halfflux_E[i - 1]);
        newtracer[i] = tracer[i] - dt / DX * (halfflux_tracer[i] - halfflux_tracer[i - 1]);
    }

    #pragma omp parallel for
    for (int i = 1; i <= N; i++) {
        rho[i] = newrho[i];
        rhov[i] = newrhov[i];
        E[i] = newE[i];
        tracer[i] = newtracer[i];
    }
}
double compute_dt() {
    double max_speed = 0.0;
    #pragma omp parallel for reduction(max:max_speed)
    for (int i = 1; i <= N; i++) {
        double speed = fabs(v[i]) + cs[i];
        if (speed > max_speed) max_speed = speed;
    }
    return 0.3 * DX / max_speed;
}
void write_output(const char *filename) {
    FILE *fp = fopen(filename, "w");
    fprintf(fp, "x,rho,v,p,e\n");
    for (int i = 1; i <= N; i++) {
        fprintf(fp, "%f,%f,%f,%f,%f\n", x[i], rho[i], v[i], p[i], e[i]);
    }
    fclose(fp);
}

void first_order_reconstruction() {
    #pragma omp parallel for
    for (int i = 0; i <= N; i++) {
        rhoL[i] = rho[i];        rhoR[i] = rho[i + 1];
        EL[i] = E[i];            ER[i] = E[i + 1];
        pL[i] = p[i];            pR[i] = p[i + 1];
        rhovL[i] = rhov[i];      rhovR[i] = rhov[i + 1];
        vL[i] = v[i];            vR[i] = v[i + 1];
        eL[i] = e[i];            eR[i] = e[i + 1];
        csL[i] = cs[i];          csR[i] = cs[i + 1];
        flux_rhoL[i] = flux_rho[i];   flux_rhoR[i] = flux_rho[i + 1];
        flux_rhovL[i] = flux_rhov[i]; flux_rhovR[i] = flux_rhov[i + 1];
        flux_EL[i] = flux_E[i];       flux_ER[i] = flux_E[i + 1];
    }
}

void hll(double dt) {
    first_order_reconstruction();

    #pragma omp parallel for
    for (int i = 0; i <= N; i++) {
        double SL, SR, pstar;
        pstar = fmax(0.0, 0.5 * (pL[i] + pR[i]) 
                          - 0.5 * (vR[i] - vL[i]) 
                          * 0.5 * (rhoL[i] + rhoR[i]) 
                          * 0.5 * (csL[i] + csR[i]));

        if (pstar <= pL[i])
            SL = vL[i] - csL[i];
        else
            SL = vL[i] - csL[i] * sqrt(1 + (GAMMA + 1) * (pstar / pL[i] - 1) / (2 * GAMMA));

        if (pstar <= pR[i])
            SR = vR[i] + csR[i];
        else
            SR = vR[i] + csR[i] * sqrt(1 + (GAMMA + 1) * (pstar / pR[i] - 1) / (2 * GAMMA));

        double tracerL = tracer[i];
        double tracerR = tracer[i + 1];
        double flux_tracerL = tracerL * vL[i];
        double flux_tracerR = tracerR * vR[i];

        if (0 <= SL) {
            halfflux_rho[i]    = flux_rhoL[i];
            halfflux_rhov[i]   = flux_rhovL[i];
            halfflux_E[i]      = flux_EL[i];
            halfflux_tracer[i] = flux_tracerL;
        } else if (SL <= 0 && 0 <= SR) {
            double inv_SR_SL = 1.0 / (SR - SL);
            halfflux_rho[i]    = (SR * flux_rhoL[i] - SL * flux_rhoR[i] + SR * SL * (rhoR[i]  - rhoL[i]))   * inv_SR_SL;
            halfflux_rhov[i]   = (SR * flux_rhovL[i] - SL * flux_rhovR[i] + SR * SL * (rhovR[i] - rhovL[i])) * inv_SR_SL;
            halfflux_E[i]      = (SR * flux_EL[i] - SL * flux_ER[i] + SR * SL * (ER[i] - EL[i]))            * inv_SR_SL;
            halfflux_tracer[i] = (SR * flux_tracerL - SL * flux_tracerR + SR * SL * (tracerR - tracerL))    * inv_SR_SL;
        } else {
            halfflux_rho[i]    = flux_rhoR[i];
            halfflux_rhov[i]   = flux_rhovR[i];
            halfflux_E[i]      = flux_ER[i];
            halfflux_tracer[i] = flux_tracerR;
        }
    }
}


void run_laxf() {
    allocate_all(1);
    setup_sod_problemA(rho, rhov, E);
    double t = 0.0;
    while (t < T_MAX) {
        boundary_problemA1st(rho, rhov, E);
        compute_observables(rho, rhov, E, cs, v, p, e);
        compute_fluxes(rho, rhov, E, v, p, flux_rho, flux_rhov, flux_E);
        double dt = compute_dt();
        lax_friedrichs_flux(dt);
        euler_step(dt);
        #pragma omp parallel for
        for (int i = 1; i <= N; i++) {
            rho[i] = newrho[i];
            rhov[i] = newrhov[i];
            E[i] = newE[i];
            v[i] = rhov[i] / rho[i];
            p[i] = (GAMMA - 1.0) * (E[i] - 0.5 * rho[i] * v[i] * v[i]);
            e[i] = E[i] / rho[i] - 0.5 * v[i] * v[i];
        }
        t += dt;
    }
    write_output("sod_lf.csv");
    free_all();
}

void run_laxw() {
    allocate_all(1);
    setup_sod_problemA(rho, rhov, E);
    double t = 0.0;
    while (t < T_MAX) {
        boundary_problemA1st(rho, rhov, E);
        compute_observables(rho, rhov, E, cs, v, p, e);
        compute_fluxes(rho, rhov, E, v, p, flux_rho, flux_rhov, flux_E);
        double dt = compute_dt();
        Lax_W_step(dt);
        #pragma omp parallel for
        for (int i = 1; i <= N; i++) {
            rho[i] = newrho[i];
            rhov[i] = newrhov[i];
            E[i] = newE[i];
            v[i] = rhov[i] / rho[i];
            p[i] = (GAMMA - 1.0) * (E[i] - 0.5 * rho[i] * v[i] * v[i]);
            e[i] = E[i] / rho[i] - 0.5 * v[i] * v[i];
        }
        t += dt;
    }
    write_output("sod_lw.csv");
    free_all();
}

void run_hll() {
    allocate_all(1);
    setup_sod_problemA(rho, rhov, E);
    double t = 0.0;
    while (t < T_MAX) {
        boundary_problemA1st(rho, rhov, E);
        compute_observables(rho, rhov, E, cs, v, p, e);
        compute_fluxes(rho, rhov, E, v, p, flux_rho, flux_rhov, flux_E);
        double dt = compute_dt();
        hll(dt);
        euler_step(dt);
        #pragma omp parallel for
        for (int i = 1; i <= N; i++) {
            rho[i] = newrho[i];
            rhov[i] = newrhov[i];
            E[i] = newE[i];
            v[i] = rhov[i] / rho[i];
            p[i] = (GAMMA - 1.0) * (E[i] - 0.5 * rho[i] * v[i] * v[i]);
            e[i] = E[i] / rho[i] - 0.5 * v[i] * v[i];
        }
        t += dt;
    }
    write_output("sod_hll.csv");
    free_all();
}

void run_hll3() {
    allocate_all(1);
    setup_sod_problemA(rho, rhov, E);
    double t = 0.0;

    FILE *ftrack = fopen("features_hll.txt", "w");
    if (!ftrack) {
        fprintf(stderr, "Error opening features_hll.txt\n");
        exit(1);
    }
    fprintf(ftrack, "t,shock_pos,rarefaction_edge,contact_pos\n");

    while (t < T_MAX) {
        boundary_problemA1st(rho, rhov, E);
        compute_observables(rho, rhov, E, cs, v, p, e);
        compute_fluxes(rho, rhov, E, v, p, flux_rho, flux_rhov, flux_E);
        double dt = compute_dt();
        hll(dt);
        euler_step(dt);

        // Update physical variables after the step
        #pragma omp parallel for
        for (int i = 1; i <= N; i++) {
            rho[i] = newrho[i];
            rhov[i] = newrhov[i];
            E[i] = newE[i];
            v[i] = rhov[i] / rho[i];
            p[i] = (GAMMA - 1.0) * (E[i] - 0.5 * rho[i] * v[i] * v[i]);
            e[i] = E[i] / rho[i] - 0.5 * v[i] * v[i];
            tracer[i] = newtracer[i];
        }

        // --- Feature Extraction (OUTSIDE PARALLEL) ---

        // Contact Discontinuity (max tracer gradient)
        double max_tracer_grad = 0.0, contact_pos = 0.0;
        for (int i = 1; i < N; i++) {
            double grad = fabs(tracer[i + 1] - tracer[i]);
            if (grad > max_tracer_grad) {
                max_tracer_grad = grad;
                contact_pos = 0.5 * (x[i] + x[i + 1]);
            }
        }


double max_rho_grad = 0.0;
double shock_pos = x[N/2];  // fallback in case no sharp gradient found

for (int i = N/2; i < N; i++) {
    double grad = fabs(rho[i+1] - rho[i]);
    if (grad > max_rho_grad) {
        max_rho_grad = grad;
        shock_pos = 0.5 * (x[i] + x[i+1]);
    }
}

        // Rarefaction Edge (first drop in rho on left)
        double rare_pos = 0.0;
        for (int i = 1; i < N / 2; i++) {
            if (fabs(rho[i] - rho[1]) > 0.01) {
                rare_pos = x[i];
                break;
            }
        }

        // Log to file
        fprintf(ftrack, "%f,%f,%f,%f\n", t, shock_pos, rare_pos, contact_pos);

        t += dt;
    }

    fclose(ftrack);
    write_output("sod_3.csv");
    free_all();
}


void run_laxf2() {
    allocate_all(1);
    setup_sod_problemB(rho, rhov, E);
    double t = 0.0;
    while (t < T_MAX2) {
        boundary_problemB1st(rho, v, p,e, E, rhov);
        compute_observables(rho, rhov, E, cs, v, p, e);
        compute_fluxes(rho, rhov, E, v, p, flux_rho, flux_rhov, flux_E);
        double dt = compute_dt();
        lax_friedrichs_flux(dt);
        euler_step(dt);
        #pragma omp parallel for
        for (int i = 1; i <= N; i++) {
            rho[i] = newrho[i];
            rhov[i] = newrhov[i];
            E[i] = newE[i];
            v[i] = rhov[i] / rho[i];
            p[i] = (GAMMA - 1.0) * (E[i] - 0.5 * rho[i] * v[i] * v[i]);
            e[i] = E[i] / rho[i] - 0.5 * v[i] * v[i];
        }
        t += dt;
    }
    write_output("sod_lf2.csv");
    free_all();
}

void run_laxw2() {
    allocate_all(1);
    setup_sod_problemB(rho, rhov, E);
    double t = 0.0;
    while (t < T_MAX2) {
        boundary_problemB1st(rho, v, p,e, E, rhov);
        compute_observables(rho, rhov, E, cs, v, p, e);
        compute_fluxes(rho, rhov, E, v, p, flux_rho, flux_rhov, flux_E);
        double dt = compute_dt();
        Lax_W_step(dt);
        #pragma omp parallel for
        for (int i = 1; i <= N; i++) {
            rho[i] = newrho[i];
            rhov[i] = newrhov[i];
            E[i] = newE[i];
            v[i] = rhov[i] / rho[i];
            p[i] = (GAMMA - 1.0) * (E[i] - 0.5 * rho[i] * v[i] * v[i]);
            e[i] = E[i] / rho[i] - 0.5 * v[i] * v[i];
        }
        t += dt;
    }
    write_output("sod_lw2.csv");
    free_all();
}

void run_hll2() {
    allocate_all(1);
    setup_sod_problemB(rho, rhov, E);
    double t = 0.0;
    while (t < T_MAX2) {
        boundary_problemB1st(rho, v, p,e, E, rhov);
        compute_observables(rho, rhov, E, cs, v, p, e);
        compute_fluxes(rho, rhov, E, v, p, flux_rho, flux_rhov, flux_E);
        double dt = compute_dt();
        hll(dt);
        euler_step(dt);
        #pragma omp parallel for
        for (int i = 1; i <= N; i++) {
            rho[i] = newrho[i];
            rhov[i] = newrhov[i];
            E[i] = newE[i];
            v[i] = rhov[i] / rho[i];
            p[i] = (GAMMA - 1.0) * (E[i] - 0.5 * rho[i] * v[i] * v[i]);
            e[i] = E[i] / rho[i] - 0.5 * v[i] * v[i];
        }
        t += dt;
    }
    write_output("sod_hll2.csv");
    free_all();
}





int main() {
    printf("Running Lax-Friedrichs...\n");
    #pragma omp parallel
    #pragma omp single
    run_laxf();

    printf("Running Lax-Wendroff...\n");
    #pragma omp parallel
    #pragma omp single
    run_laxw();

    printf("Running Hll...\n");
    #pragma omp parallel
    #pragma omp single
    run_hll();

    printf("Running Hll (with feature tracking)...\n");
    #pragma omp parallel
    #pragma omp single
    run_hll3();

    printf("Running Lax-Friedrichs2...\n");
    #pragma omp parallel
    #pragma omp single
    run_laxf2();

    printf("Running Lax-Wendroff2...\n");
    #pragma omp parallel
    #pragma omp single
    run_laxw2();

    printf("Running Hll2...\n");
    #pragma omp parallel
    #pragma omp single
    run_hll2();

    printf("Finished. Output written to 'sod_lf.csv' and 'sod_lw.csv'.\n");
    return 0;
}

