from langchain_core.documents import Document

chunk = Document(
    page_content=('항을 따릅니다. 특별약관 일반사항에서도 정하지 않은 사항은 보통약관을 따릅니다. 다만\n'
 ', 보통약관 제10조(환급금의 중도인출), 제11조(만기환급금의 지급)은 제외합니다.- 117 -# 4-4. [갱신형] 반려견 사망위로금 '
 '특별약관# 제 1조 (보험금의 지급사유)① 회사는 보험증권에 기재된 이 특별약관의 보험기간(이하「보험기간」이라 합니다) 중\n'
 '에 제3항에서 정한 보장개시일(책임개시일) 이후에 보험증권에 기재된 반려견이 보험\n'
 '기간 중에 사망한 경우 보험증권에 기재된 보험가입금액을 보험수익자에게 보상하여\n'
 '드립니다.'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000618',
              'chunk_char_len': 289,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
