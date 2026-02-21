from langchain_core.documents import Document

chunk = Document(
    page_content=("있을 때 특</td></tr></tbody></table><br><h1 id='177' style='font-size:16px'>(책임을 "
 "물을 만한 기대가능성이 없을 때)를 말하며, 가령 천재지변, 전쟁,</h1><br><p id='178' "
 "data-category='paragraph' style='font-size:14px'>사변 등으로 인해 이행이 불가능한 경우 등이 "
 '있습니다'),
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
 'indexing': {'chunk_id': 'chunk_000811',
              'chunk_char_len': 213,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
