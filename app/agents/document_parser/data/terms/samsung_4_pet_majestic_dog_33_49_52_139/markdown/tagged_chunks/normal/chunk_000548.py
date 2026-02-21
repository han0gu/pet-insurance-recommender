from langchain_core.documents import Document

chunk = Document(
    page_content=('관 일반사항에 정하지 않은 사항은 보통약관을 따릅니다. 다만, 보통약관 제10조(환급금\n'
 '의 중도인출), 제11조(만기환급금의 지급)은 제외합니다.- 109 -4-2. 반려견 의료비 확대보장 '
 '(특정처치(이물제거))(수술당일제외,\n'
 '연간2회한)(재가입형) 특별약관# 제 1조 (보험금의 지급사유)① 회사는 보험증권에 기재된 이 특별약관의 보험기간(이하「보험기간」이라 '
 '합니다) 중\n'
 '에 보험증권에 기재된 반려견이 국내에서 수의사에게 이물 섭취 치료를 목적으로 이'),
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
 'indexing': {'chunk_id': 'chunk_000548',
              'chunk_char_len': 253,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
