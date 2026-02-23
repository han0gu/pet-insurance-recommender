from langchain_core.documents import Document

chunk = Document(
    page_content=('. 또 한, 국세 및 지방세 체납시 국세청 및 지방자치단체에 의해 채무자의 해약환급 금이 압류될 수 있으며, 체납처분 절차에 따라 회사는 '
 "채권자에게 해약환급금 을 지급하게 됩니다.</td></tr></tbody></table><p id='66' "
 "data-category='paragraph' style='font-size:20px'>제 6 관 계약의 해지 및 "
 "해약환급금</p><br><p id='67' data-category='paragraph' "
 "style='font-size:20px'>등</p><h1 id='68'"),
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
 'indexing': {'chunk_id': 'chunk_000262',
              'chunk_char_len': 290,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
