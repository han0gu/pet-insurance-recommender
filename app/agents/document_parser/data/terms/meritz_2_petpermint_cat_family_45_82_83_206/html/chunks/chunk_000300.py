from langchain_core.documents import Document

chunk = Document(
    page_content=('대한 동의 거부 등<br>계약자, 피보험자 또는 보험수익자의 책임있는 사유로<br>보험금 지급사유의 조사와 확인이 지연되는 경우<br>⑥ '
 "제7항에 따라 보험금 지급사유에 대해 제3자의 의견에<br>따르기로 한 경우</p><br><h1 id='28' "
 "style='font-size:20px'>【분쟁조정 신청】</h1><br><p id='29' "
 "data-category='paragraph' style='font-size:16px'>분쟁조정 신청은 이 약관의「분쟁의 조정」조항에 "
 '따르<br>며 분쟁조정 신청 대상기관은 금융감독원의'),
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
