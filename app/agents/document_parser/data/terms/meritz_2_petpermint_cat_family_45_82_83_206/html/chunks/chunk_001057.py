from langchain_core.documents import Document

chunk = Document(
    page_content=('. 각 관절의 운동범위<br>측정은 장해평가시점의 ｢산업재해보상보험법 시행규<br>칙｣ 제47조 제1항 및 제3항의 정상인의 신체 각 '
 "관절<br>에 대한 평균 운동가능영역을 기준으로 정상각도 및<br>측정방법 등을 따른다.</p><figure id='1'><img "
 'alt="" data-coord="top-left:(278,353); bottom-right:(957,649)" '
 "/></figure><br><h1 id='2' style='font-size:20px'>< 손가락 ></h1><h1 id='3'"),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
