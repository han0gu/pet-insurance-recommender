from langchain_core.documents import Document

chunk = Document(
    page_content=('낙한 경우에 한하여 보험계약 「보험료의 납입을 연체하여 해지된 특별약관의 부활(효력\n'
 '회복)」 에 따라 이 특별약관의 부활(효력회복)을 취급합니다.제4조 (준용규정)이 특별약관에서 정하지 않은 사항은 보험계약을 따릅니다.- '
 '132 -5-3. 특별조건부 특별약관# 제 1조 (보험계약의 성립)- ① 이 특별약관은 보험계약(특별약관이 부가된 경우에는 그 특별약관을 '
 '포함합니다. 이하\n'
 '- 「보험계약」이라 합니다)을 체결할 때 피보험자의 건강상태가 보험회사(이하「회'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000709',
              'chunk_char_len': 255,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
