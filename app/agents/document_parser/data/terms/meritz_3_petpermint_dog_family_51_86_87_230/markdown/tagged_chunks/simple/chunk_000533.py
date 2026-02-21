from langchain_core.documents import Document

chunk = Document(
    page_content=('조항」에서 정하지 않은 사항은「반려동물 비용손해 관련\n'
 '특별약관 일반조항」을 따릅니다. 단,「반려동물 비용손해\n'
 '관련 특별약관 일반조항」에서 정하지 않은 사항은 보통약\n'
 '관을 따릅니다.188# Ⅲ. 기타 특별약관# 1. 자동갱신 특별약관# 제1조(특별약관의 적용)\uf000 이 특별약관은 무배당 펫퍼민트 '
 'Puppy&Family보험 다이\n'
 '렉트2601 특별약관 중 자동갱신 적용대상 특별약관(이하\n'
 '「자동갱신 적용대상 계약」이라 합니다)의 자동갱신에 대\n'
 '하여 회사와 계약자간에 합의가 되었을 경우에 적용합니다.'),
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
 'indexing': {'chunk_id': 'chunk_000533',
              'chunk_char_len': 275,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
