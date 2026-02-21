from langchain_core.documents import Document

chunk = Document(
    page_content=('약 후에 다음 각 호의 방법 중 계약자가 원하는 방법을 확인하여 지체 없이 약관 및\n'
 '계약자 보관용 청약서를 제공하여 드립니다. 만약, 회사가 전자우편 및 전자적 의사표\n'
 '시로 제공한 경우 계약자 또는 그 대리인이 약관 및 계약자 보관용 청약서 등을 수신\n'
 '하였을 때에는 해당 문서를 드린 것으로 봅니다.- 1. 서면교부\n'
 '- 2. 우편 또는 전자우편\n'
 '- 3. 휴대전화 문자메세지 또는 이에 준하는 전자적 의사표시\n'
 '② 제1항과 관련하여 통신판매계약의 경우, 회사는 계약자가 가입한 특별약관만 포함한'),
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
