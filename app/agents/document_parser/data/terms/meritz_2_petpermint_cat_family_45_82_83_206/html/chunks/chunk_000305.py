from langchain_core.documents import Document

chunk = Document(
    page_content=(". 다만, 회사는 계약자 등이 분쟁조정을 신청<br>했다는 사유만으로 이자지급을 거절하지 않습니다.</p><br><p id='36' "
 "data-category='paragraph' style='font-size:20px'>\uf000 계약자, 피보험자 또는 보험수익자는 "
 '제9조(알릴 의무<br>위반의 효과) 및 제2항의 보험금 지급사유조사와 관련하여<br>의료기관, 국민건강보험공단, 경찰서 등 관공서에 '
 '대한 회<br>사의 서면에 의한 조사요청에 동의하여야 합니다'),
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
