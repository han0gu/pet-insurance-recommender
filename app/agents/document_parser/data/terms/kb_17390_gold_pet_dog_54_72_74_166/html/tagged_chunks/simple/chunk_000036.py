from langchain_core.documents import Document

chunk = Document(
    page_content=('정신박약, 심한 등의 사물 는 의사 결정 능력이 없는 상태</td><td>의식장애 심신장애로 인하여 변별 능력 '
 "또</td></tr></tbody></table><br><h1 id='44' style='font-size:14px'>\uf000 회사는 "
 "다른 약정이 없으면 피보험자가 직업, 직무 또는 동호회 활동목적으로 아</h1><br><p id='45' "
 "data-category='paragraph' style='font-size:14px'>래에 열거된 행위로 인하여 제3조(보험금의 "
 '지급사유)의 상해 관련 보험금 지급사<br>유가 발생한'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage',
            'risk_domains': ['digestive', 'urinary']},
 'indexing': {'chunk_id': 'chunk_000036',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
