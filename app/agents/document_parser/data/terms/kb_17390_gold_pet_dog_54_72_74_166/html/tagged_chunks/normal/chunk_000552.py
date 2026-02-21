from langchain_core.documents import Document

chunk = Document(
    page_content=("정한 의료기관에서 발급한 것이어야 합니다.</p><br><table id='19' "
 "style='font-size:14px'><thead></thead><tbody><tr><td>관 련 법 규</td><td>의료법 "
 '제3조(의료기관)</td></tr><tr><td colspan="2">이 법에서 의료기관이라 함은 의료인이 공중 또는 특수 다수인을 '
 '위하여 의 료・조산의 업을 행하는 곳을 말합니다'),
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
 'indexing': {'chunk_id': 'chunk_000552',
              'chunk_char_len': 219,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
