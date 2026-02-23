from langchain_core.documents import Document

chunk = Document(
    page_content=('. \uf000 회사는 제6항의 서면조사에 대한 동의 요청시 조사목적, 사용처 등을 명시하고 '
 "설</td></tr></tbody></table><br><p id='72' data-category='paragraph' "
 "style='font-size:16px'>명합니다.</p><br><p id='73' data-category='paragraph' "
 "style='font-size:16px'>제9조(만기환급금의 지급)</p><br><h1 id='74' "
 "style='font-size:16px'>\uf000 회사는 보험기간이</h1><br><p id='75'"),
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
 'indexing': {'chunk_id': 'chunk_000058',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
