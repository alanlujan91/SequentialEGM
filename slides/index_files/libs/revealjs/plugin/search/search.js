!(function (e, t) {
  "object" == typeof exports && "undefined" != typeof module
    ? (module.exports = t())
    : "function" == typeof define && define.amd
    ? define(t)
    : ((e =
        "undefined" != typeof globalThis
          ? globalThis
          : e || self).RevealSearch = t());
})(this, function () {
  "use strict";
  var e =
      "undefined" != typeof globalThis
        ? globalThis
        : "undefined" != typeof window
        ? window
        : "undefined" != typeof global
        ? global
        : "undefined" != typeof self
        ? self
        : {},
    t = function (e) {
      try {
        return !!e();
      } catch (e) {
        return !0;
      }
    },
    n = !t(function () {
      return (
        7 !=
        Object.defineProperty({}, 1, {
          get: function () {
            return 7;
          },
        })[1]
      );
    }),
    r = function (e) {
      return e && e.Math == Math && e;
    },
    o =
      r("object" == typeof globalThis && globalThis) ||
      r("object" == typeof window && window) ||
      r("object" == typeof self && self) ||
      r("object" == typeof e && e) ||
      (function () {
        return this;
      })() ||
      Function("return this")(),
    i = t,
    c = /#|\.prototype\./,
    a = function (e, t) {
      var n = l[u(e)];
      return n == s || (n != f && ("function" == typeof t ? i(t) : !!t));
    },
    u = (a.normalize = function (e) {
      return String(e).replace(c, ".").toLowerCase();
    }),
    l = (a.data = {}),
    f = (a.NATIVE = "N"),
    s = (a.POLYFILL = "P"),
    p = a,
    d = function (e) {
      return "object" == typeof e ? null !== e : "function" == typeof e;
    },
    g = d,
    h = function (e) {
      if (!g(e)) throw TypeError(String(e) + " is not an object");
      return e;
    },
    y = d,
    v = h,
    x = function (e) {
      if (!y(e) && null !== e)
        throw TypeError("Can't set " + String(e) + " as a prototype");
      return e;
    },
    b =
      Object.setPrototypeOf ||
      ("__proto__" in {}
        ? (function () {
            var e,
              t = !1,
              n = {};
            try {
              (e = Object.getOwnPropertyDescriptor(
                Object.prototype,
                "__proto__",
              ).set).call(n, []),
                (t = n instanceof Array);
            } catch (e) {}
            return function (n, r) {
              return v(n), x(r), t ? e.call(n, r) : (n.__proto__ = r), n;
            };
          })()
        : void 0),
    m = d,
    E = b,
    S = {},
    w = d,
    O = o.document,
    R = w(O) && w(O.createElement),
    T = function (e) {
      return R ? O.createElement(e) : {};
    },
    _ =
      !n &&
      !t(function () {
        return (
          7 !=
          Object.defineProperty(T("div"), "a", {
            get: function () {
              return 7;
            },
          }).a
        );
      }),
    j = d,
    P = function (e, t) {
      if (!j(e)) return e;
      var n, r;
      if (t && "function" == typeof (n = e.toString) && !j((r = n.call(e))))
        return r;
      if ("function" == typeof (n = e.valueOf) && !j((r = n.call(e)))) return r;
      if (!t && "function" == typeof (n = e.toString) && !j((r = n.call(e))))
        return r;
      throw TypeError("Can't convert object to primitive value");
    },
    I = n,
    C = _,
    N = h,
    A = P,
    k = Object.defineProperty;
  S.f = I
    ? k
    : function (e, t, n) {
        if ((N(e), (t = A(t, !0)), N(n), C))
          try {
            return k(e, t, n);
          } catch (e) {}
        if ("get" in n || "set" in n)
          throw TypeError("Accessors not supported");
        return "value" in n && (e[t] = n.value), e;
      };
  var $ = {},
    L = function (e) {
      if (null == e) throw TypeError("Can't call method on " + e);
      return e;
    },
    M = L,
    U = function (e) {
      return Object(M(e));
    },
    D = U,
    F = {}.hasOwnProperty,
    z = function (e, t) {
      return F.call(D(e), t);
    },
    K = {}.toString,
    B = function (e) {
      return K.call(e).slice(8, -1);
    },
    W = B,
    G = "".split,
    V = t(function () {
      return !Object("z").propertyIsEnumerable(0);
    })
      ? function (e) {
          return "String" == W(e) ? G.call(e, "") : Object(e);
        }
      : Object,
    Y = L,
    q = function (e) {
      return V(Y(e));
    },
    X = Math.ceil,
    H = Math.floor,
    J = function (e) {
      return isNaN((e = +e)) ? 0 : (e > 0 ? H : X)(e);
    },
    Q = J,
    Z = Math.min,
    ee = function (e) {
      return e > 0 ? Z(Q(e), 9007199254740991) : 0;
    },
    te = J,
    ne = Math.max,
    re = Math.min,
    oe = q,
    ie = ee,
    ce = function (e, t) {
      var n = te(e);
      return n < 0 ? ne(n + t, 0) : re(n, t);
    },
    ae = function (e) {
      return function (t, n, r) {
        var o,
          i = oe(t),
          c = ie(i.length),
          a = ce(r, c);
        if (e && n != n) {
          for (; c > a; ) if ((o = i[a++]) != o) return !0;
        } else
          for (; c > a; a++)
            if ((e || a in i) && i[a] === n) return e || a || 0;
        return !e && -1;
      };
    },
    ue = { includes: ae(!0), indexOf: ae(!1) },
    le = {},
    fe = z,
    se = q,
    pe = ue.indexOf,
    de = le,
    ge = function (e, t) {
      var n,
        r = se(e),
        o = 0,
        i = [];
      for (n in r) !fe(de, n) && fe(r, n) && i.push(n);
      for (; t.length > o; ) fe(r, (n = t[o++])) && (~pe(i, n) || i.push(n));
      return i;
    },
    he = [
      "constructor",
      "hasOwnProperty",
      "isPrototypeOf",
      "propertyIsEnumerable",
      "toLocaleString",
      "toString",
      "valueOf",
    ].concat("length", "prototype");
  $.f =
    Object.getOwnPropertyNames ||
    function (e) {
      return ge(e, he);
    };
  var ye = { exports: {} },
    ve = function (e, t) {
      return {
        enumerable: !(1 & e),
        configurable: !(2 & e),
        writable: !(4 & e),
        value: t,
      };
    },
    xe = S,
    be = ve,
    me = n
      ? function (e, t, n) {
          return xe.f(e, t, be(1, n));
        }
      : function (e, t, n) {
          return (e[t] = n), e;
        },
    Ee = o,
    Se = me,
    we = function (e, t) {
      try {
        Se(Ee, e, t);
      } catch (n) {
        Ee[e] = t;
      }
      return t;
    },
    Oe = we,
    Re = "__core-js_shared__",
    Te = o[Re] || Oe(Re, {}),
    _e = Te;
  (ye.exports = function (e, t) {
    return _e[e] || (_e[e] = void 0 !== t ? t : {});
  })("versions", []).push({
    version: "3.12.1",
    mode: "global",
    copyright: "© 2021 Denis Pushkarev (zloirock.ru)",
  });
  var je,
    Pe,
    Ie = 0,
    Ce = Math.random(),
    Ne = function (e) {
      return (
        "Symbol(" +
        String(void 0 === e ? "" : e) +
        ")_" +
        (++Ie + Ce).toString(36)
      );
    },
    Ae = o,
    ke = o,
    $e = function (e) {
      return "function" == typeof e ? e : void 0;
    },
    Le = function (e, t) {
      return arguments.length < 2
        ? $e(Ae[e]) || $e(ke[e])
        : (Ae[e] && Ae[e][t]) || (ke[e] && ke[e][t]);
    },
    Me = Le("navigator", "userAgent") || "",
    Ue = o.process,
    De = Ue && Ue.versions,
    Fe = De && De.v8;
  Fe
    ? (Pe = (je = Fe.split("."))[0] < 4 ? 1 : je[0] + je[1])
    : Me &&
      (!(je = Me.match(/Edge\/(\d+)/)) || je[1] >= 74) &&
      (je = Me.match(/Chrome\/(\d+)/)) &&
      (Pe = je[1]);
  var ze = Pe && +Pe,
    Ke = t,
    Be =
      !!Object.getOwnPropertySymbols &&
      !Ke(function () {
        return !String(Symbol()) || (!Symbol.sham && ze && ze < 41);
      }),
    We = Be && !Symbol.sham && "symbol" == typeof Symbol.iterator,
    Ge = o,
    Ve = ye.exports,
    Ye = z,
    qe = Ne,
    Xe = Be,
    He = We,
    Je = Ve("wks"),
    Qe = Ge.Symbol,
    Ze = He ? Qe : (Qe && Qe.withoutSetter) || qe,
    et = function (e) {
      return (
        (Ye(Je, e) && (Xe || "string" == typeof Je[e])) ||
          (Xe && Ye(Qe, e) ? (Je[e] = Qe[e]) : (Je[e] = Ze("Symbol." + e))),
        Je[e]
      );
    },
    tt = d,
    nt = B,
    rt = et("match"),
    ot = h,
    it = function () {
      var e = ot(this),
        t = "";
      return (
        e.global && (t += "g"),
        e.ignoreCase && (t += "i"),
        e.multiline && (t += "m"),
        e.dotAll && (t += "s"),
        e.unicode && (t += "u"),
        e.sticky && (t += "y"),
        t
      );
    },
    ct = {},
    at = t;
  function ut(e, t) {
    return RegExp(e, t);
  }
  (ct.UNSUPPORTED_Y = at(function () {
    var e = ut("a", "y");
    return (e.lastIndex = 2), null != e.exec("abcd");
  })),
    (ct.BROKEN_CARET = at(function () {
      var e = ut("^r", "gy");
      return (e.lastIndex = 2), null != e.exec("str");
    }));
  var lt = { exports: {} },
    ft = Te,
    st = Function.toString;
  "function" != typeof ft.inspectSource &&
    (ft.inspectSource = function (e) {
      return st.call(e);
    });
  var pt,
    dt,
    gt,
    ht = ft.inspectSource,
    yt = ht,
    vt = o.WeakMap,
    xt = "function" == typeof vt && /native code/.test(yt(vt)),
    bt = ye.exports,
    mt = Ne,
    Et = bt("keys"),
    St = xt,
    wt = d,
    Ot = me,
    Rt = z,
    Tt = Te,
    _t = function (e) {
      return Et[e] || (Et[e] = mt(e));
    },
    jt = le,
    Pt = "Object already initialized",
    It = o.WeakMap;
  if (St || Tt.state) {
    var Ct = Tt.state || (Tt.state = new It()),
      Nt = Ct.get,
      At = Ct.has,
      kt = Ct.set;
    (pt = function (e, t) {
      if (At.call(Ct, e)) throw new TypeError(Pt);
      return (t.facade = e), kt.call(Ct, e, t), t;
    }),
      (dt = function (e) {
        return Nt.call(Ct, e) || {};
      }),
      (gt = function (e) {
        return At.call(Ct, e);
      });
  } else {
    var $t = _t("state");
    (jt[$t] = !0),
      (pt = function (e, t) {
        if (Rt(e, $t)) throw new TypeError(Pt);
        return (t.facade = e), Ot(e, $t, t), t;
      }),
      (dt = function (e) {
        return Rt(e, $t) ? e[$t] : {};
      }),
      (gt = function (e) {
        return Rt(e, $t);
      });
  }
  var Lt = {
      set: pt,
      get: dt,
      has: gt,
      enforce: function (e) {
        return gt(e) ? dt(e) : pt(e, {});
      },
      getterFor: function (e) {
        return function (t) {
          var n;
          if (!wt(t) || (n = dt(t)).type !== e)
            throw TypeError("Incompatible receiver, " + e + " required");
          return n;
        };
      },
    },
    Mt = o,
    Ut = me,
    Dt = z,
    Ft = we,
    zt = ht,
    Kt = Lt.get,
    Bt = Lt.enforce,
    Wt = String(String).split("String");
  (lt.exports = function (e, t, n, r) {
    var o,
      i = !!r && !!r.unsafe,
      c = !!r && !!r.enumerable,
      a = !!r && !!r.noTargetGet;
    "function" == typeof n &&
      ("string" != typeof t || Dt(n, "name") || Ut(n, "name", t),
      (o = Bt(n)).source ||
        (o.source = Wt.join("string" == typeof t ? t : ""))),
      e !== Mt
        ? (i ? !a && e[t] && (c = !0) : delete e[t],
          c ? (e[t] = n) : Ut(e, t, n))
        : c
        ? (e[t] = n)
        : Ft(t, n);
  })(Function.prototype, "toString", function () {
    return ("function" == typeof this && Kt(this).source) || zt(this);
  });
  var Gt = Le,
    Vt = S,
    Yt = n,
    qt = et("species"),
    Xt = n,
    Ht = o,
    Jt = p,
    Qt = function (e, t, n) {
      var r, o;
      return (
        E &&
          "function" == typeof (r = t.constructor) &&
          r !== n &&
          m((o = r.prototype)) &&
          o !== n.prototype &&
          E(e, o),
        e
      );
    },
    Zt = S.f,
    en = $.f,
    tn = function (e) {
      var t;
      return tt(e) && (void 0 !== (t = e[rt]) ? !!t : "RegExp" == nt(e));
    },
    nn = it,
    rn = ct,
    on = lt.exports,
    cn = t,
    an = Lt.enforce,
    un = function (e) {
      var t = Gt(e),
        n = Vt.f;
      Yt &&
        t &&
        !t[qt] &&
        n(t, qt, {
          configurable: !0,
          get: function () {
            return this;
          },
        });
    },
    ln = et("match"),
    fn = Ht.RegExp,
    sn = fn.prototype,
    pn = /a/g,
    dn = /a/g,
    gn = new fn(pn) !== pn,
    hn = rn.UNSUPPORTED_Y;
  if (
    Xt &&
    Jt(
      "RegExp",
      !gn ||
        hn ||
        cn(function () {
          return (
            (dn[ln] = !1), fn(pn) != pn || fn(dn) == dn || "/a/i" != fn(pn, "i")
          );
        }),
    )
  ) {
    for (
      var yn = function (e, t) {
          var n,
            r = this instanceof yn,
            o = tn(e),
            i = void 0 === t;
          if (!r && o && e.constructor === yn && i) return e;
          gn
            ? o && !i && (e = e.source)
            : e instanceof yn && (i && (t = nn.call(e)), (e = e.source)),
            hn && (n = !!t && t.indexOf("y") > -1) && (t = t.replace(/y/g, ""));
          var c = Qt(gn ? new fn(e, t) : fn(e, t), r ? this : sn, yn);
          hn && n && (an(c).sticky = !0);
          return c;
        },
        vn = function (e) {
          (e in yn) ||
            Zt(yn, e, {
              configurable: !0,
              get: function () {
                return fn[e];
              },
              set: function (t) {
                fn[e] = t;
              },
            });
        },
        xn = en(fn),
        bn = 0;
      xn.length > bn;

    )
      vn(xn[bn++]);
    (sn.constructor = yn), (yn.prototype = sn), on(Ht, "RegExp", yn);
  }
  un("RegExp");
  var mn = {},
    En = {},
    Sn = {}.propertyIsEnumerable,
    wn = Object.getOwnPropertyDescriptor,
    On = wn && !Sn.call({ 1: 2 }, 1);
  En.f = On
    ? function (e) {
        var t = wn(this, e);
        return !!t && t.enumerable;
      }
    : Sn;
  var Rn = n,
    Tn = En,
    _n = ve,
    jn = q,
    Pn = P,
    In = z,
    Cn = _,
    Nn = Object.getOwnPropertyDescriptor;
  mn.f = Rn
    ? Nn
    : function (e, t) {
        if (((e = jn(e)), (t = Pn(t, !0)), Cn))
          try {
            return Nn(e, t);
          } catch (e) {}
        if (In(e, t)) return _n(!Tn.f.call(e, t), e[t]);
      };
  var An = {};
  An.f = Object.getOwnPropertySymbols;
  var kn = $,
    $n = An,
    Ln = h,
    Mn =
      Le("Reflect", "ownKeys") ||
      function (e) {
        var t = kn.f(Ln(e)),
          n = $n.f;
        return n ? t.concat(n(e)) : t;
      },
    Un = z,
    Dn = Mn,
    Fn = mn,
    zn = S,
    Kn = o,
    Bn = mn.f,
    Wn = me,
    Gn = lt.exports,
    Vn = we,
    Yn = function (e, t) {
      for (var n = Dn(t), r = zn.f, o = Fn.f, i = 0; i < n.length; i++) {
        var c = n[i];
        Un(e, c) || r(e, c, o(t, c));
      }
    },
    qn = p,
    Xn = it,
    Hn = ct,
    Jn = ye.exports,
    Qn = RegExp.prototype.exec,
    Zn = Jn("native-string-replace", String.prototype.replace),
    er = Qn,
    tr = (function () {
      var e = /a/,
        t = /b*/g;
      return (
        Qn.call(e, "a"), Qn.call(t, "a"), 0 !== e.lastIndex || 0 !== t.lastIndex
      );
    })(),
    nr = Hn.UNSUPPORTED_Y || Hn.BROKEN_CARET,
    rr = void 0 !== /()??/.exec("")[1];
  (tr || rr || nr) &&
    (er = function (e) {
      var t,
        n,
        r,
        o,
        i = this,
        c = nr && i.sticky,
        a = Xn.call(i),
        u = i.source,
        l = 0,
        f = e;
      return (
        c &&
          (-1 === (a = a.replace("y", "")).indexOf("g") && (a += "g"),
          (f = String(e).slice(i.lastIndex)),
          i.lastIndex > 0 &&
            (!i.multiline || (i.multiline && "\n" !== e[i.lastIndex - 1])) &&
            ((u = "(?: " + u + ")"), (f = " " + f), l++),
          (n = new RegExp("^(?:" + u + ")", a))),
        rr && (n = new RegExp("^" + u + "$(?!\\s)", a)),
        tr && (t = i.lastIndex),
        (r = Qn.call(c ? n : i, f)),
        c
          ? r
            ? ((r.input = r.input.slice(l)),
              (r[0] = r[0].slice(l)),
              (r.index = i.lastIndex),
              (i.lastIndex += r[0].length))
            : (i.lastIndex = 0)
          : tr && r && (i.lastIndex = i.global ? r.index + r[0].length : t),
        rr &&
          r &&
          r.length > 1 &&
          Zn.call(r[0], n, function () {
            for (o = 1; o < arguments.length - 2; o++)
              void 0 === arguments[o] && (r[o] = void 0);
          }),
        r
      );
    });
  var or = er;
  (function (e, t) {
    var n,
      r,
      o,
      i,
      c,
      a = e.target,
      u = e.global,
      l = e.stat;
    if ((n = u ? Kn : l ? Kn[a] || Vn(a, {}) : (Kn[a] || {}).prototype))
      for (r in t) {
        if (
          ((i = t[r]),
          (o = e.noTargetGet ? (c = Bn(n, r)) && c.value : n[r]),
          !qn(u ? r : a + (l ? "." : "#") + r, e.forced) && void 0 !== o)
        ) {
          if (typeof i == typeof o) continue;
          Yn(i, o);
        }
        (e.sham || (o && o.sham)) && Wn(i, "sham", !0), Gn(n, r, i, e);
      }
  })({ target: "RegExp", proto: !0, forced: /./.exec !== or }, { exec: or });
  var ir = lt.exports,
    cr = h,
    ar = t,
    ur = it,
    lr = "toString",
    fr = RegExp.prototype,
    sr = fr.toString,
    pr = ar(function () {
      return "/a/b" != sr.call({ source: "a", flags: "b" });
    }),
    dr = sr.name != lr;
  (pr || dr) &&
    ir(
      RegExp.prototype,
      lr,
      function () {
        var e = cr(this),
          t = String(e.source),
          n = e.flags;
        return (
          "/" +
          t +
          "/" +
          String(
            void 0 === n && e instanceof RegExp && !("flags" in fr)
              ? ur.call(e)
              : n,
          )
        );
      },
      { unsafe: !0 },
    );
  var gr = lt.exports,
    hr = or,
    yr = t,
    vr = et,
    xr = me,
    br = vr("species"),
    mr = RegExp.prototype,
    Er = !yr(function () {
      var e = /./;
      return (
        (e.exec = function () {
          var e = [];
          return (e.groups = { a: "7" }), e;
        }),
        "7" !== "".replace(e, "$<a>")
      );
    }),
    Sr = "$0" === "a".replace(/./, "$0"),
    wr = vr("replace"),
    Or = !!/./[wr] && "" === /./[wr]("a", "$0"),
    Rr = !yr(function () {
      var e = /(?:)/,
        t = e.exec;
      e.exec = function () {
        return t.apply(this, arguments);
      };
      var n = "ab".split(e);
      return 2 !== n.length || "a" !== n[0] || "b" !== n[1];
    }),
    Tr = J,
    _r = L,
    jr = function (e) {
      return function (t, n) {
        var r,
          o,
          i = String(_r(t)),
          c = Tr(n),
          a = i.length;
        return c < 0 || c >= a
          ? e
            ? ""
            : void 0
          : (r = i.charCodeAt(c)) < 55296 ||
            r > 56319 ||
            c + 1 === a ||
            (o = i.charCodeAt(c + 1)) < 56320 ||
            o > 57343
          ? e
            ? i.charAt(c)
            : r
          : e
          ? i.slice(c, c + 2)
          : o - 56320 + ((r - 55296) << 10) + 65536;
      };
    },
    Pr = { codeAt: jr(!1), charAt: jr(!0) }.charAt,
    Ir = U,
    Cr = Math.floor,
    Nr = "".replace,
    Ar = /\$([$&'`]|\d{1,2}|<[^>]*>)/g,
    kr = /\$([$&'`]|\d{1,2})/g,
    $r = B,
    Lr = or,
    Mr = function (e, t, n, r) {
      var o = vr(e),
        i = !yr(function () {
          var t = {};
          return (
            (t[o] = function () {
              return 7;
            }),
            7 != ""[e](t)
          );
        }),
        c =
          i &&
          !yr(function () {
            var t = !1,
              n = /a/;
            return (
              "split" === e &&
                (((n = {}).constructor = {}),
                (n.constructor[br] = function () {
                  return n;
                }),
                (n.flags = ""),
                (n[o] = /./[o])),
              (n.exec = function () {
                return (t = !0), null;
              }),
              n[o](""),
              !t
            );
          });
      if (
        !i ||
        !c ||
        ("replace" === e && (!Er || !Sr || Or)) ||
        ("split" === e && !Rr)
      ) {
        var a = /./[o],
          u = n(
            o,
            ""[e],
            function (e, t, n, r, o) {
              var c = t.exec;
              return c === hr || c === mr.exec
                ? i && !o
                  ? { done: !0, value: a.call(t, n, r) }
                  : { done: !0, value: e.call(n, t, r) }
                : { done: !1 };
            },
            {
              REPLACE_KEEPS_$0: Sr,
              REGEXP_REPLACE_SUBSTITUTES_UNDEFINED_CAPTURE: Or,
            },
          ),
          l = u[0],
          f = u[1];
        gr(String.prototype, e, l),
          gr(
            mr,
            o,
            2 == t
              ? function (e, t) {
                  return f.call(e, this, t);
                }
              : function (e) {
                  return f.call(e, this);
                },
          );
      }
      r && xr(mr[o], "sham", !0);
    },
    Ur = h,
    Dr = ee,
    Fr = J,
    zr = L,
    Kr = function (e, t, n) {
      return t + (n ? Pr(e, t).length : 1);
    },
    Br = function (e, t, n, r, o, i) {
      var c = n + e.length,
        a = r.length,
        u = kr;
      return (
        void 0 !== o && ((o = Ir(o)), (u = Ar)),
        Nr.call(i, u, function (i, u) {
          var l;
          switch (u.charAt(0)) {
            case "$":
              return "$";
            case "&":
              return e;
            case "`":
              return t.slice(0, n);
            case "'":
              return t.slice(c);
            case "<":
              l = o[u.slice(1, -1)];
              break;
            default:
              var f = +u;
              if (0 === f) return i;
              if (f > a) {
                var s = Cr(f / 10);
                return 0 === s
                  ? i
                  : s <= a
                  ? void 0 === r[s - 1]
                    ? u.charAt(1)
                    : r[s - 1] + u.charAt(1)
                  : i;
              }
              l = r[f - 1];
          }
          return void 0 === l ? "" : l;
        })
      );
    },
    Wr = function (e, t) {
      var n = e.exec;
      if ("function" == typeof n) {
        var r = n.call(e, t);
        if ("object" != typeof r)
          throw TypeError(
            "RegExp exec method returned something other than an Object or null",
          );
        return r;
      }
      if ("RegExp" !== $r(e))
        throw TypeError("RegExp#exec called on incompatible receiver");
      return Lr.call(e, t);
    },
    Gr = Math.max,
    Vr = Math.min;
  Mr("replace", 2, function (e, t, n, r) {
    var o = r.REGEXP_REPLACE_SUBSTITUTES_UNDEFINED_CAPTURE,
      i = r.REPLACE_KEEPS_$0,
      c = o ? "$" : "$0";
    return [
      function (n, r) {
        var o = zr(this),
          i = null == n ? void 0 : n[e];
        return void 0 !== i ? i.call(n, o, r) : t.call(String(o), n, r);
      },
      function (e, r) {
        if ((!o && i) || ("string" == typeof r && -1 === r.indexOf(c))) {
          var a = n(t, e, this, r);
          if (a.done) return a.value;
        }
        var u = Ur(e),
          l = String(this),
          f = "function" == typeof r;
        f || (r = String(r));
        var s = u.global;
        if (s) {
          var p = u.unicode;
          u.lastIndex = 0;
        }
        for (var d = []; ; ) {
          var g = Wr(u, l);
          if (null === g) break;
          if ((d.push(g), !s)) break;
          "" === String(g[0]) && (u.lastIndex = Kr(l, Dr(u.lastIndex), p));
        }
        for (var h, y = "", v = 0, x = 0; x < d.length; x++) {
          g = d[x];
          for (
            var b = String(g[0]),
              m = Gr(Vr(Fr(g.index), l.length), 0),
              E = [],
              S = 1;
            S < g.length;
            S++
          )
            E.push(void 0 === (h = g[S]) ? h : String(h));
          var w = g.groups;
          if (f) {
            var O = [b].concat(E, m, l);
            void 0 !== w && O.push(w);
            var R = String(r.apply(void 0, O));
          } else R = Br(b, l, m, E, w, r);
          m >= v && ((y += l.slice(v, m) + R), (v = m + b.length));
        }
        return y + l.slice(v);
      },
    ];
  });
  var Yr = {};
  Yr[et("toStringTag")] = "z";
  var qr = "[object z]" === String(Yr),
    Xr = qr,
    Hr = B,
    Jr = et("toStringTag"),
    Qr =
      "Arguments" ==
      Hr(
        (function () {
          return arguments;
        })(),
      ),
    Zr = Xr
      ? Hr
      : function (e) {
          var t, n, r;
          return void 0 === e
            ? "Undefined"
            : null === e
            ? "Null"
            : "string" ==
              typeof (n = (function (e, t) {
                try {
                  return e[t];
                } catch (e) {}
              })((t = Object(e)), Jr))
            ? n
            : Qr
            ? Hr(t)
            : "Object" == (r = Hr(t)) && "function" == typeof t.callee
            ? "Arguments"
            : r;
        },
    eo = qr
      ? {}.toString
      : function () {
          return "[object " + Zr(this) + "]";
        },
    to = qr,
    no = lt.exports,
    ro = eo;
  to || no(Object.prototype, "toString", ro, { unsafe: !0 });
  /*!
   * Handles finding a text string anywhere in the slides and showing the next occurrence to the user
   * by navigatating to that slide and highlighting it.
   *
   * @author Jon Snyder <snyder.jon@gmail.com>, February 2013
   */
  return function () {
    var e, t, n, r, o, i, c;
    function a() {
      (t = document.createElement("div")).classList.add("searchbox"),
        (t.style.position = "absolute"),
        (t.style.top = "10px"),
        (t.style.right = "10px"),
        (t.style.zIndex = 10),
        (t.innerHTML =
          '<input type="search" class="searchinput" placeholder="Search..." style="vertical-align: top;"/>\n\t\t</span>'),
        ((n = t.querySelector(".searchinput")).style.width = "240px"),
        (n.style.fontSize = "14px"),
        (n.style.padding = "4px 6px"),
        (n.style.color = "#000"),
        (n.style.background = "#fff"),
        (n.style.borderRadius = "2px"),
        (n.style.border = "0"),
        (n.style.outline = "0"),
        (n.style.boxShadow = "0 2px 18px rgba(0, 0, 0, 0.2)"),
        (n.style["-webkit-appearance"] = "none"),
        e.getRevealElement().appendChild(t),
        n.addEventListener(
          "keyup",
          function (t) {
            switch (t.keyCode) {
              case 13:
                t.preventDefault(),
                  (function () {
                    if (i) {
                      var t = n.value;
                      "" === t
                        ? (c && c.remove(), (r = null))
                        : ((c = new f("slidecontent")),
                          (r = c.apply(t)),
                          (o = 0));
                    }
                    r &&
                      (r.length && r.length <= o && (o = 0),
                      r.length > o && (e.slide(r[o].h, r[o].v), o++));
                  })(),
                  (i = !1);
                break;
              default:
                i = !0;
            }
          },
          !1,
        ),
        l();
    }
    function u() {
      t || a(), (t.style.display = "inline"), n.focus(), n.select();
    }
    function l() {
      t || a(), (t.style.display = "none"), c && c.remove();
    }
    function f(t, n) {
      var r = document.getElementById(t) || document.body,
        o = n || "EM",
        i = new RegExp("^(?:" + o + "|SCRIPT|FORM)$"),
        c = ["#ff6", "#a0ffff", "#9f9", "#f99", "#f6f"],
        a = [],
        u = 0,
        l = "",
        f = [];
      (this.setRegex = function (e) {
        (e = e.replace(/^[^\w]+|[^\w]+$/g, "").replace(/[^\w'-]+/g, "|")),
          (l = new RegExp("(" + e + ")", "i"));
      }),
        (this.getRegex = function () {
          return l
            .toString()
            .replace(/^\/\\b\(|\)\\b\/i$/g, "")
            .replace(/\|/g, " ");
        }),
        (this.hiliteWords = function (t) {
          if (null != t && t && l && !i.test(t.nodeName)) {
            if (t.hasChildNodes())
              for (var n = 0; n < t.childNodes.length; n++)
                this.hiliteWords(t.childNodes[n]);
            var r, s;
            if (3 == t.nodeType)
              if ((r = t.nodeValue) && (s = l.exec(r))) {
                for (var p = t; null != p && "SECTION" != p.nodeName; )
                  p = p.parentNode;
                var d = e.getIndices(p),
                  g = f.length,
                  h = !1;
                for (n = 0; n < g; n++)
                  f[n].h === d.h && f[n].v === d.v && (h = !0);
                h || f.push(d),
                  a[s[0].toLowerCase()] ||
                    (a[s[0].toLowerCase()] = c[u++ % c.length]);
                var y = document.createElement(o);
                y.appendChild(document.createTextNode(s[0])),
                  (y.style.backgroundColor = a[s[0].toLowerCase()]),
                  (y.style.fontStyle = "inherit"),
                  (y.style.color = "#000");
                var v = t.splitText(s.index);
                (v.nodeValue = v.nodeValue.substring(s[0].length)),
                  t.parentNode.insertBefore(y, v);
              }
          }
        }),
        (this.remove = function () {
          for (
            var e, t = document.getElementsByTagName(o);
            t.length && (e = t[0]);

          )
            e.parentNode.replaceChild(e.firstChild, e);
        }),
        (this.apply = function (e) {
          if (null != e && e)
            return this.remove(), this.setRegex(e), this.hiliteWords(r), f;
        });
    }
    return {
      id: "search",
      init: function (n) {
        (e = n).registerKeyboardShortcut("CTRL + Shift + F", "Search"),
          document.addEventListener(
            "keydown",
            function (e) {
              "F" == e.key &&
                (e.ctrlKey || e.metaKey) &&
                (e.preventDefault(),
                t || a(),
                "inline" !== t.style.display ? u() : l());
            },
            !1,
          );
      },
      open: u,
    };
  };
});
