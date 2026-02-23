from langchain_core.documents import Document

chunk = Document(
    page_content=('경우보다 위험이 높은 반려동물이 가입하기 위<br>한 방법의 하나로, 특정 질병 또는 특정 부위를 보장에<br>서 제외하는 방법을 '
 "말합니다.</p><h1 id='88' style='font-size:20px'>【 보험금 삭감 】</h1><br><p id='89' "
 "data-category='paragraph' style='font-size:20px'>일반적인 경우보다 위험이 높은 반려동물이 가입하기 "
 '위<br>한 방법의 하나로, 보험 가입 후 기간이 경과함에 따라<br>위험의 크기 및 정도가 점차 감소하는 위험에 대해 적용<br>하여'),
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
