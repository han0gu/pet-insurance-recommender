from langchain_core.documents import Document

chunk = Document(
    page_content=('경과함에 따라<br>위험의 크기 및 정도가 점차 감소하는 위험에 대해 적용<br>하여 보험 가입 후 일정기간 내에 보험사고가 발생할 '
 "경<br>우 미리 정해진 비율로 보험금을 감액하여 지급하는 방<br>법을 말합니다.</p><h1 id='90' "
 "style='font-size:16px'>【 보험료 할증 】</h1><br><p id='91' "
 "data-category='paragraph' style='font-size:16px'>일반적인 경우보다 위험이 높은 반려동물이 가입하기 "
 '위<br>한 방법의 하나로, 보험 가입 후 기간이 경과함에'),
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
