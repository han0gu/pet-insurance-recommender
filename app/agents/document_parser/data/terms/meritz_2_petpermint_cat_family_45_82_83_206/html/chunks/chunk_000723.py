from langchain_core.documents import Document

chunk = Document(
    page_content=("이전에 이미 감염 또는 발병<br>한 질병 및 상해</p><footer id='31' "
 "style='font-size:14px'>149</footer><p id='32' data-category='list' "
 "style='font-size:16px'>⑦ 원인이 어떠한 경우에도 반려동물에 대한 사료제공 또<br>는 급수 등 기본적인 관리에 대한 "
 '태만<br>⑧ 반려동물을 범죄행위, 경주, 수색, 폭약탐지, 구조,<br>실험 및 이와 유사한 목적으로 이용함으로써 '
 '발생한<br>손해<br>⑨ 수의사의 치료상의 과오로 생긴 상해 또는 질병,'),
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
