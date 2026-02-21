from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 제1항에도 불구하고 국가유공자 등 예우 및 지원에 관한 법률에 따른 상이자의 증\n'
 '명을 받은 사람 또는 장애인복지법에 따른 장애인등록증을 발급받은 사람에 대해서\n'
 '는 해당 증명서·장애인등록증의 사본이나 그 밖의 장애 사실을 증명하는 서류를\n'
 '반\n'
 '제출하는 경우에는 제 1항의 장애인증명서는 제출하지 않을 수 있습니다.\n'
 '려\n'
 '\uf000 장애인으로서 그 장애기간이 기재된 장애인증명서를 제1항 따라 회사에 제출한 때 동\n'
 '에는 그 장애기간 동안은 이를 다시 제출하지 않을 수 있습니다. 물'),
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
 'indexing': {'chunk_id': 'chunk_000808',
              'chunk_char_len': 263,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
