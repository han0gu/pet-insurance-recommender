from langchain_core.documents import Document

chunk = Document(
    page_content=('기타지급금을 합한 금액이 1인당 "1억원까지"(본 보험회사의 여타 보호상품과 합산) 보호됩니다. 이와 별도로\n'
 '본 보험회사 보호상품의 사고보험금을 합산한 금액이 1인당 "1억원까지" 보호됩니다. 다만, 보험계약자 및 보\n'
 '험료납부자가 법인인 보험계약의 경우에는 보호되지 않습니다.- 20 -당신에게 좋은보험 삼성화재반려견보험 애니펫특별약관- 21 -당신에게 '
 '좋은보험 삼성화재# 수술비용 확대보장 특별약관# 제1조(보상하는 손해)- ① 회사는 보험기간 중에 보험증권에 기재된 반려동물에게 '
 '보통약관에서 보상하는 상해 또는 질병이'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000094',
              'chunk_char_len': 291,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
