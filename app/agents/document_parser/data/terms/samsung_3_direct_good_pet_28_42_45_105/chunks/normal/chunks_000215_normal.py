from langchain_core.documents import Document

chunk = Document(
    page_content=('④ 회사는 이 특별약관의 청약을 받고 제1회 보험료를 받은 경우에 건강진단을 받지 않 는 특별약관은 청약일, 진단계약은 진단일(재진단의 '
 '경우에는 최종진단일)부터 30일 이내에 승낙 또는 거절하여야 하며, 승낙한 때에는 보험증권을 드립니다. 그러나 30일 이내에 승낙 또는 '
 '거절의 통지가 없으면 승낙된 것으로 봅니다. ⑤ 회사가 제1회 보험료를 받고 승낙을 거절한 경우에는 거절통지와 함께 받은 금액을 '
 '계약자에게 돌려 드리며, 보험료를 받은 기간에 대하여 평균공시이율+1%를 연단위 복리로 계산한 금액을 더하여 지급합니다'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 51},
 'term_type': 'special',
 'clause': {'clause_type': 'renewal', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000215',
              'chunk_char_len': 290,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.85}},
)
