from langchain_core.documents import Document

chunk = Document(
    page_content=(': 주사기 등으로 빨아들이는 것 ∙ 천자 : 바늘 또는 관을 꽂아 체액․조직을 뽑아내거나 약물을 주입하는 '
 "것</td></tr></tbody></table><h1 id='241' "
 "style='font-size:16px'>제5조(특별약관의</h1><br><p id='242' "
 "data-category='paragraph' style='font-size:16px'>소멸)</p><br><p id='243' "
 "data-category='paragraph' style='font-size:16px'>피보험자가 사망하였을 경우에는 이 특별약관"),
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
 'indexing': {'chunk_id': 'chunk_000681',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
