from langchain_core.documents import Document

chunk = Document(
    page_content=('- 집니다.\n'
 '<용어풀이># [현저하게 공정을 잃은 합의]회사가 보험수익자의 경제적․신체적․정신적인 어려움, 경솔함, 경험 부족 등을 이용하여 '
 '동일․유사\n'
 '사례에 비추어 보험수익자에게 매우 불합리하게 합의를 하는 것을 의미합니다.# 제 27조 (특별약관의 재가입에 관한 사항)① 계약이 다음 '
 '각 호의 조건을 충족하고 계약자가 제4항에 따라 재가입의사를 표시한\n'
 '때에는 특별약관 일반사항의 제19조(특별약관의 성립) 및 제21조(약관 교부 및 설명\n'
 '의무 등)를 준용하여 회사가 정한 절차에 따라 계약자는 기존 계약에 이어 재가입할'),
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
 'indexing': {'chunk_id': 'chunk_000538',
              'chunk_char_len': 292,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
