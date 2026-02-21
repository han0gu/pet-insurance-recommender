from langchain_core.documents import Document

chunk = Document(
    page_content=('따른 해약환급금을 계약자에게 지급합니다.\n'
 '\uf000 계약자는 제1항에 따른 제척기간에도 불구하고 민법 등 관계 법령에서 정하는 바- \n'
 '에 따라 법률상의 권리를 행사할 수 있습니다.# 용 어 풀이∙위법계약\n'
 '위법계약이라 함은 ｢금융소비자보호에 관한 법률｣ 제47조에서 정한 적합성원\n'
 '칙, 적정성원칙, 설명의무, 불공정영업행위 금지 또는 부당권유행위 금지를 위- \n'
 '# 반한 계약을# 말합니다.∙ 제척기간'),
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
 'indexing': {'chunk_id': 'chunk_000179',
              'chunk_char_len': 218,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
