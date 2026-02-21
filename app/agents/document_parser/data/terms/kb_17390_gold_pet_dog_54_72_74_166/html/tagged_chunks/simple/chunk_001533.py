from langchain_core.documents import Document

chunk = Document(
    page_content=('모습)을 남긴 때’라 함은 상처의 흔적, 화상 등으로 피부의<br>변색, 모발의 결손, 조직(뼈, 피부 등)의 결손 및 함몰 등으로 '
 '성형수술<br>을 하여도 더 이상 추상(추한 모습)이 없어지지 않는 경우를 말한다.<br>4) 다발성 반흔(흉터) 발생시 각 '
 '판정부위(얼굴, 머리, 목) 내의 다발성 반<br>흔(흉터)의 길이 또는 면적은 합산하여 평가한다'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other',
            'risk_domains': ['digestive', 'head', 'skin']},
 'indexing': {'chunk_id': 'chunk_001533',
              'chunk_char_len': 198,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
