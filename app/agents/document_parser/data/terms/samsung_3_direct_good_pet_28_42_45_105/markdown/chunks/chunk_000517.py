from langchain_core.documents import Document

chunk = Document(
    page_content=('서도 정하지 않은 사항은 보통약관을 따릅니다.- 92 -3-7. [갱신형] 반려견 위탁비용(반려인질병입원1 일이상180일한도)(실손) '
 '특별약관# 제1조 (보험금의 지급사유)- ① 회사는 보험증권에 기재된 피보험자가 이 특별약관의 보험기간(이하 「보험기간」 이라\n'
 '- 합니다) 중에 진단확정된 질병으로 병원 또는 의원(한방병원 또는 한의원을 포함합니\n'
 '- 다)에 1일이상 계속 입원하여 치료를 받은 경우에는 입원기간 동안 보험증권에 기재\n'
 '- 된 반려견을 수탁기관에 위탁함으로써 발생한 위탁비용을 반려견 위탁비용으로 보험'),
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
