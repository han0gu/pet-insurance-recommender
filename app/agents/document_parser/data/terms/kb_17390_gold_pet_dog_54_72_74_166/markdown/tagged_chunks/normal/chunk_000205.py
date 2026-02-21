from langchain_core.documents import Document

chunk = Document(
    page_content=('| 관 련 법 규 | 민법 제2조(신의성실) 제1항 |\n'
 '| ① 권리의 행사와 의무의 이행은 신의에 좇아 성실히 하여야 한다. | ① 권리의 행사와 의무의 이행은 신의에 좇아 성실히 하여야 '
 '한다. |\n'
 '# 제47조(설명서 교부 및 보험안내자료 등의 효력)| \uf000 회사는 일반금융소비자에게 청약을 권유하거나 |\n'
 '| --- |\n'
 '일반금융소비자가 설명을 요청하는 경우 보험상품에 관한 중요한 사항을 계약자가 이해할 수 있도록 설명하고 계\n'
 '약자가 이해하였음을 서명(전자서명법 제2조 제2호에 따른 전자서명을 포함), 기'),
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
 'indexing': {'chunk_id': 'chunk_000205',
              'chunk_char_len': 282,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
