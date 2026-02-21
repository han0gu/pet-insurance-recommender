from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:14px'>장해판정기준</p><br><p id='9' data-category='paragraph' "
 "style='font-size:14px'>1) ‘외모’란 얼굴(눈, 코, 귀, 입 포함), 머리, 목을 말한다.<br>2) ‘추상(추한 "
 '모습)장해’라 함은 성형수술(반흔(흉터)성형술, 레이저치<br>료 등 포함)을 시행한 후에도 영구히 남게 되는 상태의 추상(추한 '
 '모습)<br>을 말한다.<br>3) ‘추상(추한 모습)을 남긴 때’라 함은 상처의 흔적, 화상 등으로 피부의<br>변색, 모발의 결손,'),
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
            'risk_domains': ['digestive', 'eye', 'head', 'skin']},
 'indexing': {'chunk_id': 'chunk_001532',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
