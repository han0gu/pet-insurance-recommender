from langchain_core.documents import Document

chunk = Document(
    page_content=('니다. 이와 별도로 본 보험회사 보호상품의 사고보험금\n'
 '을 합산한 금액이 1인당 “1억원까지” 보호됩니다. 다\n'
 '만, 계약자 및 보험료납부자가 법인인 보험계약의 경우\n'
 '에는 보호되지 않습니다.86무배당 펫퍼민트 Puppy&Family보험\n'
 '다이렉트2601 특별약관8788# Ⅰ. 반려동물 비용손해 관련 특별약관반려동물 비용손해 관련 특별약관 일반조항# 제1조(목적)이 '
 '특별약관은 계약자와 회사 사이에 피보험자 소유의 보험\n'
 '증권에 기재된 반려동물의 질병 또는 상해로 인한 손해를'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000142',
              'chunk_char_len': 262,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
