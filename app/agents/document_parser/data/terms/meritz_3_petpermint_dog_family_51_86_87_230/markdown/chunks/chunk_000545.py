from langchain_core.documents import Document

chunk = Document(
    page_content=('기간 중에 회사가 지정한 질병(이하「특정질병」이라 합니\n'
 '다)(【별첨(특정질병 분류표(반려견))】)을 직접적인 원인\n'
 '으로 계약에서 정한 보험금 지급사유가 발생한 경우에는 회\n'
 '사는 보험금을 지급하지 않습니다.\n'
 '\uf000 제1항의 회사가 보험금을 지급하지 않는 기간(이하 「부\n'
 '담보 기간」이라 합니다)은 특정질병의 상태에 따라「1개월\n'
 '부터 5년」또는 「계약의 보험기간」(단, 계약이 갱신 또는\n'
 '재가입 계약인 경우 최초 계약일로부터 최종 갱신 또는 재\n'
 '가입 계약의 종료일까지의 기간을 말하며, 이하 「계약의'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
