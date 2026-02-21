from langchain_core.documents import Document

chunk = Document(
    page_content=('신됩니다.# 제7조 (준용규정)이 특별약관에 정하지 않은 사항은 특별약관 일반사항을 따릅니다. 특별약관 일반사항에\n'
 '서도 정하지 않은 사항은 보통약관을 따릅니다. 다만, 보통약관 제10조(환급금의 중도인\n'
 '출), 제11조(만기환급금의 지급)은 제외합니다.- 125 -4-7. [갱신형] 반려견 위탁비용(반려인질병입원1일이상180일한도)(실손) '
 '특별약관# 제1조 (보험금의 지급사유)- ① 회사는 보험증권에 기재된 피보험자가 이 특별약관의 보험기간(이하 「보험기간」 이라'),
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
 'indexing': {'chunk_id': 'chunk_000674',
              'chunk_char_len': 260,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
