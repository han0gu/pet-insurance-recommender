from langchain_core.documents import Document

chunk = Document(
    page_content=('수두폐렴(J17.1*)]</td><td>B01.2+</td></tr><tr><td>[B05.2+: 폐렴이 합병된 '
 '홍역(J17.1*)]</td><td>B05.2+</td></tr><tr><td>[B25.0+: '
 '거대세포바이러스폐렴(J17.1*)]</td><td>B25.0+</td></tr><tr><td>[B58.3+: 폐 '
 '톡소포자충증(J17.3*)]</td><td>B58.3+</td></tr><tr><td>재향군인병</td><td>A48.1</td></tr></tbody></table><br><p '
 "id='103'"),
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
 'indexing': {'chunk_id': 'chunk_001783',
              'chunk_char_len': 287,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
