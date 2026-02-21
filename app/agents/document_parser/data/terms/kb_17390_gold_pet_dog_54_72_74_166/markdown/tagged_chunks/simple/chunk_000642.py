from langchain_core.documents import Document

chunk = Document(
    page_content=('118 KB 금쪽같은 펫보험(강아지)(무배당)(26.01)영업을 행하는 시설을 말합니다.\n'
 '\uf000 제1항의 "장례서비스"라 함은 화장서비스, 장례서비스, 필수 장례용품 등을 말하\n'
 '며, 장례 이전 반려동물 사체 임시 안치, 한지·자개 등 기능성 유골함 및 수목\n'
 '함, 유골보석 제작, 봉안당(납골당) 안치는 포함하지 않습니다.\n'
 '\uf000 제1항의 반려동물장례비용지원금은 총 장례비용에 보험증권에 기재된 보상비율을\n'
 '곱한 금액이며 보험증권에 기재된 보험가입금액을 한도로 합니다.\n'
 '\uf000 제1항의 경우 반려동물장례비용지원금보장개시일은 계약일로부터 그날을 포함하'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000642',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
