from langchain_core.documents import Document

chunk = Document(
    page_content=('체결한 후 보장개시일 이전에 동일한 특정질병이 발생한 경\n'
 '우에는 계약을 무효로 하지 않습니다.# 제2조(특별면책조건의 내용)\uf000 이 특별약관에서 정한 회사가 보험금을 지급하지 않는\n'
 '기간 중에 회사가 지정한 질병(이하「특정질병」이라 합니\n'
 '다)(【별첨(특정질병 분류표(반려묘))】)을 직접적인 원인\n'
 '으로 계약에서 정한 보험금 지급사유가 발생한 경우에는 회\n'
 '사는 보험금을 지급하지 않습니다.\uf000 제1항의 회사가 보험금을 지급하지 않는 기간(이하 「부\n'
 '담보 기간」이라 합니다)은 특정질병의 상태에 따라「1개월'),
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
