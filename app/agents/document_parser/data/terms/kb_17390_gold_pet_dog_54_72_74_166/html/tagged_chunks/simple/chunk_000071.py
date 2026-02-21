from langchain_core.documents import Document

chunk = Document(
    page_content=('. ∙ 운용자산이익률 사업방법서에 정한 방법에 따라 운용자산수익률에서 투자지출률을 차감하여 산 '
 "출합니다.</td></tr></tbody></table><p id='86' data-category='paragraph' "
 "style='font-size:14px'>제10조(보험금 받는 방법의</p><br><p id='87' "
 "data-category='paragraph' style='font-size:14px'>변경)</p><br><p id='88' "
 "data-category='list' style='font-size:14px'>\uf000"),
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
 'indexing': {'chunk_id': 'chunk_000071',
              'chunk_char_len': 294,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
