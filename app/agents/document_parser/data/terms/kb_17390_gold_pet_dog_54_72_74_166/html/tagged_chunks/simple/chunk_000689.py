from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 아래와 같이<br>관<br>환경성질환입원일당이 지급된 최종입원일부터 180일이 경과하도록 퇴원없이 계속</p><br><h1 '
 "id='252' style='font-size:16px'>입원중인 경우에는 입원일당이 지급된 최종입원일의 그 다음날을 퇴원일로 "
 '봅니다.</h1><br><figure id=\'253\'><img style=\'font-size:14px\' alt="예 시'),
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
 'indexing': {'chunk_id': 'chunk_000689',
              'chunk_char_len': 212,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
