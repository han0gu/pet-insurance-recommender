from langchain_core.documents import Document

chunk = Document(
    page_content=('. 또한 원인과 질환에 따라 동시에 사용될 수 있습니다. - 검표(+) : 원인이 되는 질환에 대한 '
 "질병분류코드</td></tr></tbody></table><br><p id='6' data-category='paragraph' "
 "style='font-size:16px'>- 별표(*) : 원인(검표)으로 인한 발현증세에 대한 질병분류코드</p><table "
 "id='7' style='font-size:16px'><thead><tr><td>3"),
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
 'indexing': {'chunk_id': 'chunk_000011',
              'chunk_char_len': 245,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
