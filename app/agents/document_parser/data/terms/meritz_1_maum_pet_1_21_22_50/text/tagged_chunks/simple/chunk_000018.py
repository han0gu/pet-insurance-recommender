from langchain_core.documents import Document

chunk = Document(
    page_content=('로서 조금만 주의를 하였다면 충분히 피해의 발생을 막을 수 있었음에도 그 주의\n'
 '조차 태만히 한 높은 강도의 주의의무 위반(이하 같습니다.)2. 지진, 분화, 해일, 홍수 또는 이와 유사한 자연재해로 생긴 손해\n'
 '3. 전쟁, 외국의 무력행사, 혁명, 내란, 사변, 폭동, 소요, 기타 이들과 유사한 사태\n'
 '4. 핵연료물질 또는 핵연료물질에 의하여 오염된 물질의 방사성, 폭발성, 그 밖의 유해\n'
 '한 특성 또는 이들의 특성에 의한 사고【핵연료물질】사용된 연료를 포함합니다.'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000018',
              'chunk_char_len': 259,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
