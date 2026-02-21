from langchain_core.documents import Document

chunk = Document(
    page_content=('국가, 지방자치단체, 한국은행, 금융회사, 주권상장법인 등을 포함하며 「금융소비자 보호에 관한\n'
 '법률」제2조(정의) 제9호에서 정하는 전문금융소비자를 말합니다.# [일반금융소비자]\n'
 '전문금융소비자가 아닌 계약자를 말합니다.② 제1항에도 불구하고 청약한 날부터 30일(다만, 65세 이상의 계약자가 전화를 이용하\n'
 '여 계약을 체결한 경우 45일)이 초과된 계약은 청약을 철회할 수 없습니다.<관련법규>[금융소비자보호에 관한 법률 제46조(청약의 '
 '철회)에서 정한 청약철회가능 기간]'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000187',
              'chunk_char_len': 264,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
