from langchain_core.documents import Document

chunk = Document(
    page_content=('. 제3조(준용규정) 134 KB 금쪽같은 '
 "펫보험(강아지)(무배당)(26.01)</td></tr></tbody></table><br><h1 id='10' "
 "style='font-size:14px'>이 초회보험료자동납입 추가특별약관에 정하지 않은 사항은 보통약관 및 "
 "보험료자동</h1><br><h1 id='11' style='font-size:14px'>납입 특별약관을 따릅니다.</h1><h1 "
 "id='12' style='font-size:20px'>4"),
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
 'indexing': {'chunk_id': 'chunk_001365',
              'chunk_char_len': 251,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
