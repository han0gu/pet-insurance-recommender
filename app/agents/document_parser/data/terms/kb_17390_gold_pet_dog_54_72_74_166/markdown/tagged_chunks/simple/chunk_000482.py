from langchain_core.documents import Document

chunk = Document(
    page_content=('성\n'
 '∙ 상법 제651조의2(서면에 의한 질문의 효력)\n'
 '특\n'
 '보험자가 서면으로 질문한 사항은 중요한 사항으로 추정한다. 약- 103 -KB 금쪽같은 펫보험(강아지)(무배당)(26.01) '
 '103병질질병려동물# 제8조(계약 후 알릴 의무)- \uf000 계약자 또는 피보험자는 보험기간 중에 피보험자에게 다음 각 호의 변경이 '
 '발생\n'
 '- 한 경우에는 우편, 전화, 방문 등의 방법으로 지체없이 회사에 알려야 합니다.\n'
 '- 1. 청약서의 기재사항을 변경하고자 할 때 또는 변경이 생겼음을 알았을 때'),
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
 'indexing': {'chunk_id': 'chunk_000482',
              'chunk_char_len': 265,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
