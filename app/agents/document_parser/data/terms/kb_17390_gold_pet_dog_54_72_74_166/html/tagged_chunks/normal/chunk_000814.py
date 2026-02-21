from langchain_core.documents import Document

chunk = Document(
    page_content=('. 피보험자가 부담한 치료비</td><td rowspan="2">×</td><td>다른</td></tr><tr><td>계약이 없을 때 이 '
 '계약의 지급보험금 다른계약이 없는 것으로 하여 각각 계산한 지급보험금의 '
 '합계액</td></tr></tbody></table><br><figure id=\'183\'><img alt="" '
 'data-coord="top-left:(830,380); bottom-right:(1354,485)" /></figure><br><p '
 "id='184' data-category='paragraph'"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000814',
              'chunk_char_len': 288,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
