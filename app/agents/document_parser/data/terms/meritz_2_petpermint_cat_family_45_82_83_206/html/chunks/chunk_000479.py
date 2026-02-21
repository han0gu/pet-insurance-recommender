from langchain_core.documents import Document

chunk = Document(
    page_content=("산출방식】</p><br><p id='77' data-category='paragraph' style='font-size:20px'>보험금 "
 "지급금액 = [(피보험자가 부담한 치료비－자기부담금)<br>× 보상비율, 지급 한도액] 중 적은 금액</p><p id='78' "
 "data-category='paragraph' style='font-size:20px'>【보험금 지급금액(자기부담금 3만원인 "
 "경우)[예시]】</p><br><p id='79' data-category='paragraph' "
 "style='font-size:20px'>① 입원"),
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
