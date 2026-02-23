from langchain_core.documents import Document

chunk = Document(
    page_content=('. 회사는 전자문서가<br>수신되지 않은 것을 확인한 경우에는 서면(등기우편 등)으<br>로 다시 알려드립니다.<br>\uf000 제1항 '
 '제2호에 따른 계약의 해지가 보험금 지급사유 발<br>생 후에 이루어진 경우에는 제16조(상해보험계약 후 알릴<br>의무) 제4항 또는 '
 '제5항에 따라 보험금을 지급합니다.<br>\uf000 제1항에도 불구하고 알릴 의무를 위반한 사실이 보험금<br>지급사유 발생에 영향을 '
 '미쳤음을 회사가 증명하지 못한 경<br>우에는 제4항 및 제5항에 관계없이 약정한 보험금을 지급합<br>니다.<br>\uf000 회사는 '
 '다른 보험가입내역에'),
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
