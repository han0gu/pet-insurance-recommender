from langchain_core.documents import Document

chunk = Document(
    page_content=('- 관을 부가하는 경우, 사망보험금을 지급할 때 피보험자의 법정상속인이 아닌 자가\n'
 '- 청구하는 경우 법정상속인의 확인서 등)\n'
 '② 제1항 제2호의 사고증명서는 의료법 제3조(의료기관)에서 규정한 국내의 병원이나 의\n'
 '원 또는 국외의 의료관련법에서 정한 의료기관에서 발급한 것이어야 합니다.<관련법규>[의료법 제3조(의료기관)]# 제8조 (보험금의 '
 '지급절차)- ① 회사는 제7조(보험금의 청구)에서 정한 서류를 접수한 때에는 접수증을 드리고 휴대전\n'
 '- 화 문자메시지 또는 전자우편 등으로 송부하며, 그 서류를 접수한 날부터 3영업일 이'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
