from langchain_core.documents import Document

chunk = Document(
    page_content=('. 이 경우 상시고용된 수의사의 범위, 신고방법, 처방전 발급 및 보존 방법, 진료부 작성 및 보고, 교육, 준수사항 등 그 밖에 필요한 '
 "사항은 농림축산식품부령으로 정한다.</td></tr></tbody></table><h1 id='38' "
 "style='font-size:14px'>제6조(수술의 정의와</h1><br><p id='39' "
 "data-category='paragraph' style='font-size:14px'>장소)</p><br><p id='40' "
 "data-category='paragraph'"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_001066',
              'chunk_char_len': 284,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
