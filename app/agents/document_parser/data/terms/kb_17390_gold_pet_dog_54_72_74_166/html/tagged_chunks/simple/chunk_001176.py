from langchain_core.documents import Document

chunk = Document(
    page_content=(". 이 계약과 다른 계약이 모두 의무보험인 경우에도 같습니다.</p><br><p id='190' "
 "data-category='paragraph' style='font-size:14px'>이 계약의 보상책임액<br>손해액 "
 '×<br>다른계약이 없는 것으로 하여 각각 계산한 보상책임액의 합계액</p><br><figure id=\'191\'><img alt="" '
 'data-coord="top-left:(823,561); bottom-right:(1419,616)" /></figure><br><p '
 "id='192'"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_001176',
              'chunk_char_len': 280,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
