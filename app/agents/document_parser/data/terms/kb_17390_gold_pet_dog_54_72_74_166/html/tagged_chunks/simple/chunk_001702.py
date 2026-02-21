from langchain_core.documents import Document

chunk = Document(
    page_content=("보장하는 상병 해당 여부를 다시 판단하지 않습니다.</p><table id='34' "
 "style='font-size:14px'><thead></thead><tbody><tr><td></td><td>보</td></tr></tbody></table><br><p "
 "id='35' data-category='paragraph' style='font-size:16px'>별표4 "
 '골절분류표Ⅱ(치아파절제외)<br>통약<br>\uf000 약관에 규정하는 골절로 분류되는 상병은 제9차 개정 '
 '한국표준질병․사인분류(KCD,<br>관<br>통계청 고시'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion',
            'risk_domains': ['dental', 'digestive']},
 'indexing': {'chunk_id': 'chunk_001702',
              'chunk_char_len': 290,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
