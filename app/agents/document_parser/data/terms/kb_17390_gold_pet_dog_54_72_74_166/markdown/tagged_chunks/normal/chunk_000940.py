from langchain_core.documents import Document

chunk = Document(
    page_content=('- 인정하지 않는다.\n'
 '- 타) 각종 기질성 정신장해와 외상후 뇌전증에 한하여 보상한다.\n'
 '- 파) 외상후 스트레스장애, 우울증(반응성) 등의 질환, 정신분열증(조현\n'
 '- 병), 편집증, 조울증(양극성장애), 불안장애, 전환장애, 공포장애,\n'
 '- 강박장애 등 각종 신경증 및 각종 인격장애는 보상의 대상이 되지\n'
 '- 않는다.\n'
 '3) 치매- \n'
 '- 가) “치매”라 함은 정상적으로 성숙한 뇌가 질병이나 외상 후 기질성\n'
 '- 손상으로 파괴되어 한번 획득한 지적기능이 지속적 또는 전반적으로\n'
 '- 저하되는 것을 말한다.'),
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
            'risk_domains': ['digestive', 'head', 'urinary']},
 'indexing': {'chunk_id': 'chunk_000940',
              'chunk_char_len': 278,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
