from langchain_core.documents import Document

chunk = Document(
    page_content=('| --- |\n'
 '| 관 련 법 규 전자서명법 제2조(정의) "전자서명" 이란 다음 각 목의 사항을 나타내는 데 이용하기 위하여 전자문서 에 첨부되거나 '
 '논리적으로 결합된 전자적형태의 정보를 말한다. 가. 서명자의 신원 |\n'
 '# 나. 서명자가 해당 전자문서에 서명하였다는 사실제21조(계약의 무효)\n'
 '다음 중 한 가지에 해당되는 경우에는 계약을 무효로 하며 이미 납입한 보험료를 돌\n'
 '려드립니다. 다만, 회사의 고의 또는 과실로 계약이 무효로 된 경우와 회사가 승낙\n'
 '전에 무효임을 알았거나 알 수 있었음에도 보험료를 반환하지 않은 경우에는 보험료'),
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
 'indexing': {'chunk_id': 'chunk_000118',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
