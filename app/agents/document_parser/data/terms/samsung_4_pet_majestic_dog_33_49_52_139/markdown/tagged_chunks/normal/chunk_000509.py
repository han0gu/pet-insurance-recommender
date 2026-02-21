from langchain_core.documents import Document

chunk = Document(
    page_content=('계약체결일부터 5년 이내(사기사실을 안 날부터 1개월 이내)에 계약을 취소할 수 있습니\n'
 '다.# 제 16조 (특별약관의 무효)계약을 체결할 때 계약에서 정한 반려견의 나이에 미달되었거나 초과되었을 경우 이 특\n'
 '별약관은 무효로 하며 이미 납입한 이 특별약관의 보험료를 돌려드립니다. 다만, 회사의\n'
 '고의 또는 과실로 특별약관이 무효로 된 경우와 회사가 승낙 전에 무효임을 알았거나 알\n'
 '수 있었음에도 불구하고 보험료를 반환하지 않은 경우에는 보험료를 납입한 날의 다음날'),
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
 'indexing': {'chunk_id': 'chunk_000509',
              'chunk_char_len': 258,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
