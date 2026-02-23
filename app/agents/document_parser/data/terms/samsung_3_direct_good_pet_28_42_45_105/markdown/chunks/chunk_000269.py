from langchain_core.documents import Document

chunk = Document(
    page_content=('- 약관 제33조(해약환급금)을 적용합니다.\n'
 '제2관 개별사항- \n'
 '# 제1조 (보험금의 지급사유)회사는 피보험자가 보험증권에 기재된 이 특별약관의 보험기간(이하「보험기간」이라 합\n'
 '니다) 중에 상해의 직접적인 결과로써 사망한 경우(질병으로 인한 사망은 제외합니다) 5\n'
 '년간 매월 보험증권에 기재된 이 특별약관의 보험가입금액을 보험금 지급사유 발생일(단,\n'
 '해당월에 보험금 지급사유 발생일이 없는 경우에는 해당월의 마지막 날로 합니다)에 반려'),
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
