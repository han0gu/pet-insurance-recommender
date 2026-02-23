from langchain_core.documents import Document

chunk = Document(
    page_content=('- 끝나는 날의 다음날부터 1개월 이내에 계약자가 계약의 해지를 청구한 때에는 회사는\n'
 '- 보험료의 자동대출납입이 없었던 것으로 하여 제35조(해약환급금) 제1항에 따른 해약\n'
 '- 환급금을 지급합니다.\n'
 '- ⑤ 회사는 자동대출납입이 종료된 날부터 15일 이내에 자동대출납입이 종료되었음을 서\n'
 '- 면, 전화(음성녹음) 또는 전자문서(SMS 포함) 등으로 계약자에게 안내하여 드립니다.'),
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
